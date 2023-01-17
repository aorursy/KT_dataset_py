from selenium import webdriver

from selenium.webdriver.firefox.options import Options

from lxml import etree

import xml.etree

import requests as req

from bs4 import BeautifulSoup

from pathlib import Path

import pandas as pd

import json

from fastai.text import *

output_folder = Path.home() / "data"



output_raw_path = output_folder / "raw_essays.json"

output_path = output_folder / "essays.csv"

output_lite = output_folder / "essays_lite.csv"



uri_initial = "https://educacao.uol.com.br/bancoderedacoes/"



# xpaths for getting the relevant contents



xpath_see_more_button = "//div[6]/section/div/section/div/div[2]/div/button"

xpath_theme_link = "//div[6]/section/div/section/div/div/div/div/a/@href"



xpath_theme_title = "//div[5]/article/div[1]/header/h1/span/i[2]/text()"

xpath_theme_text = '//div[5]/article/div[2]/div/div[1]/div/div[2]/div[1]//div[@class="text"]'

xpath_essay_link = "//article/div[2]/div/div[1]/div/div[2]/div[2]/section/article/div/a/@href"



xpath_essay_title = "//div[5]/article/div[2]/div/div[1]/div/div[2]/div[1]/div[2]/section/div[2]/h2/text()"

xpath_essay_title_alt = '//*[@id="conteudo-principal"]/header/h1/text()'

xpath_essay_text = "//div[5]/article/div[2]/div/div[1]/div/div[2]/div[1]/div[2]/section/div[2]/div[2]"

xpath_essay_text_alt = '//*[@id="texto"]'

xpath_essay_scores = "//div[5]/article/div[2]/div/div[1]/div/div[2]/div[1]/div[4]/section[1]/article/div/span[2]/text()"

xpath_essay_scores_alt = '//*[@id="texto"]/div[1]/table[1]//tr/td[2]/text()'
options = Options()

options.headless = True

driver = webdriver.Firefox(options=options)

driver.get(uri_initial)

while True:

    try:

        elem = driver.find_element_by_xpath(xpath_see_more_button)

        elem.click()

    except:

        break



page_source = driver.page_source

driver.close()
html_parser = etree.HTMLParser()

tree = etree.fromstring(page_source, html_parser)

uris_theme = tree.xpath(xpath_theme_link)
def get_clean_theme_text(raw_text):

    soup = BeautifulSoup(raw_text, 'html.parser')

    return soup.text





def get_theme(uri):

    r = req.get(uri)

    page_source = r.text    

    tree = etree.fromstring(page_source, html_parser)

    

    title = tree.xpath(xpath_theme_title)[0]

    text_el = tree.xpath(xpath_theme_text)[0]

    raw_text= etree.tostring(text_el, encoding=str)

    text = get_clean_theme_text(raw_text)

    uri_essays = tree.xpath(xpath_essay_link)

    

    theme_obj = {"title": title,

                 "raw_text": raw_text,

                 "clean_text": text,

                 "url": uri,

                 "urls_essay": uri_essays}

    

    return theme_obj
themes = []



for (i, uri) in enumerate(uris_theme):

    print("\r{}/{}   ".format(i, len(uris_theme)), end="")

    themes.append(get_theme(uri))
def get_clean_essay_text(raw_text, alt=False):

    soup = BeautifulSoup(raw_text, 'html.parser')

    if alt is not False:

        all_spans = soup.findAll("span")

        black_spans = soup.findAll("span", {"style":"color:black"})

        non_black_spans_set = set(all_spans) - set(black_spans)

        non_black_spans = [span for span in all_spans if span in non_black_spans_set]

        for span in non_black_spans:

            span.decompose()

    text = soup.text

    return text





def get_essay(url):

    r = req.get(url)

    page_source = r.text

    tree = etree.fromstring(page_source, html_parser)

    essay_obj = {}

    try:

        el = tree.xpath(xpath_essay_text)[0]

        title = tree.xpath(xpath_essay_title)[0]

        scores = tree.xpath(xpath_essay_scores)

    except:

        el = tree.xpath(xpath_essay_text_alt)[0]

        title = tree.xpath(xpath_essay_title_alt)[0]

        scores = tree.xpath(xpath_essay_scores_alt)

    finally:

        raw_text = etree.tostring(el, encoding=str)

        text = get_clean_essay_text(raw_text)        

        essay_obj = {"title": title,

                     "raw_text": raw_text,

                     "clean_text": text,

                     "scores": scores,

                     "url": url}

        return essay_obj
for i, theme in enumerate(themes):

    theme_essays = []

    urls_essay = theme["urls_essay"]

    for j, uri_essay in enumerate(urls_essay):

        print("\r{}/{} ({}/{})       ".format(i, len(themes), j, len(urls_essay)), end="")

        essay = get_essay(uri_essay)

        theme_essays.append(essay)

    theme_url = theme["url"]

    theme["essays"] = theme_essays
with open(output_raw_path, "w") as fid:

    json.dump(themes, fid)
dataset = []





for theme in themes:

    essays = theme["essays"]

    theme_title = theme["title"]

    theme_text = theme["clean_text"]

    for essay in essays:

        row = {}

        row["essay_title"] = essay["title"]

        row["essay_text"] = essay["clean_text"]

        row["scores"] = essay["scores"]

        row["theme_title"] = theme_title

        row["theme_text"] = theme_text

        dataset.append(row)
data = pd.DataFrame(dataset)
score_map = {0: "score_1", 1: "score_2", 2:"score_3", 3:"score_4", 4:"score_5", 5: "total_score"}

scores = (data.scores.apply(pd.Series)

              .rename(columns=score_map))



data = (data.drop("scores", axis=1)

            .join(scores))
data.to_csv(output_path)
data = (pd.read_csv(output_path)

          .assign(score_1=lambda x: pd.to_numeric(x["score_1"], errors='coerce'))

          .assign(score_2=lambda x: pd.to_numeric(x["score_2"], errors='coerce'))

          .assign(score_3=lambda x: pd.to_numeric(x["score_3"], errors='coerce'))

          .assign(score_4=lambda x: pd.to_numeric(x["score_4"], errors='coerce'))

          .assign(score_5=lambda x: pd.to_numeric(x["score_5"], errors='coerce'))

          .assign(score_5=lambda x: pd.to_numeric(x["score_5"], errors='coerce'))

          .assign(total_score=lambda x: pd.to_numeric(x["total_score"], errors='coerce'))) 
low_data = data.loc[:, ["essay_text", "total_score"]].dropna()

low_data.total_score = (low_data.total_score > 500).astype(int)

low_data.to_csv(output_lite, index=False)
#even_low_data = low_data.loc[:, ["essay_text"]]

#even_low_data.to_csv(output_folder / "essays_even_liter.csv", index=False)
data_lm = TextClasDataBunch.from_csv(output_folder,

                                     "essays_lite.csv",

                                     text_cols=["essay_text"],

                                     label_cols=["total_score"],

                                     bs=48)

data_lm.show_batch()

data_lm.backwards = False
torch.cuda.empty_cache()
learn = text_classifier_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
txt = """

Recentemente, o Governo Federal anunciou um contingenciamento de verbas para a educação superior, o que gerou grande divergência entre os brasileiros: enquanto uns apoiaram a medida, outros foram às ruas para se manifestar contra ela. Nesse contexto, alguns defenderam a privatização das universidades ou a cobrança de mensalidades para os mais ricos a fim de solucionar a crise financeira do ensino superior. Porém, o mais justo é manter esse serviço público e gratuito a todos, embora certas ações sejam necessárias para maximizar os benefícios dessa decisão.



Primordialmente Primeiramente, é válido importante considerar que as universidades públicas são as mais bem avaliadas do país. Segundo uma pesquisa do grupo britânico QS (Quacquarelli Symonds), UNICAMP e USP, por exemplo, estão entre as melhores da América Latina. Ademais, um levantamento recente da CAPES (Coordenação de Aperfeiçoamento de Pessoal de Nível Superior) mostrou que mais de 90% das pesquisas científicas do Brasil vêm das escolas superiores públicas. Sendo assim, a privatização das universidades seria prejudicial ao país, já que sua qualidade tenderia a piorar e a quantidade dessas pesquisas, a diminuir. Além disso, seria provável que tais estudos, lamentavelmente, fossem voltados somente a áreas que trazem maior retorno financeiro imediato. Desse modo, conclui-se que as instituições públicas de ensino superior devem continuar sendo do Estado.



Em uma segunda análise, convém lembrar que o Artigo 206 da Constituição Federal garante o ensino gratuito em estabelecimentos oficiais. Logo, ninguém deveria ser privado de ter uma educação superior gratuita e de qualidade, independentemente da de classe social. Infelizmente, isso não acontece na prática: boa parte da população, sobretudo os mais pobres, não tem acesso a esse serviço. Todavia, a cobrança de mensalidades não seria uma solução adequada para o impasse, pois não só violaria a Constituição, mas também poderia desestimular os mais ricos a ingressarem e realizarem pesquisas nessas instituições. Assim, é possível concluir que a universidade pública deve ser gratuita para todos.



Portanto, a privatização das universidades públicas e a cobrança da mensalidade dos mais ricos não solucionariam o problema da crise financeira no ensino superior. Contudo, o Poder Público precisa agir para resolvê-lo e proporcionar, ainda, uma maior igualdade de acesso a essas instituições. Primeiramente, verbas de setores menos relevantes (a exemplo das Forças Armadas) devem ser destinadas a áreas mais essenciais, como a educação superior. Além do mais, para esse mesmo fim, deve haver privatizações de empresas estatais (os Correios, por exemplo). Por fim, para que haja maior igualdade de oportunidades nas universidades, é necessário que as cotas sociais (àqueles que estudaram em escolas públicas) sejam ampliadas. Dessa forma, a crise financeira do ensino superior será solucionada de maneira justa e em respeito à Constituição Federal.

"""



learn.predict(txt)