# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# https://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html

import random

import numba

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import style
import seaborn as sns
# sns.set(context="notebook", style="darkgrid", palette="hls") # sns.set(context="notebook", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.8)
# style.use("ggplot")
# style.use("classic")

import os
def create_dir_if_doesnt_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

%matplotlib inline
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
df_original = pd.read_csv('../input/backup_prefeitura.tar.xz', parse_dates=['data_inicio_atividade', 'data_emissao', 'data_expiracao'])
df_original = df_original[
    df_original['latitude'].notnull() &
    df_original['longitude'].notnull() &
    df_original['numero_do_alvara'].notnull() &
    (df_original['data_inicio_atividade'] > '1980-01-01')
]
df_original = df_original[
#     (df_original['atividade_principal_agregada'] != 'X')
    df_original['atividade_principal'].notnull()
]
'ok'
df_tmp = df_original[df_original['bairro'] == 'BATEL'].copy().replace(np.nan, '', regex=True)
df_tmp['atividade_caracteristicas'] = (
    ' ' + (
        df_tmp['atividade_principal'].str.replace(".*outr[ao]s.*n[aã]o especif.*", '', regex=True) \
            .str.replace(".*divers[ao]s? n[aã]o especif.*", '', regex=True) \
            .str.lower()
      ) + ' '
)
# df_tmp['atividade_caracteristicas'] = (
#     ' ' + df_tmp['atividade_principal'].str.replace(".*outr[ao]s.*n[aã]o especif.*", '', regex=True).str.lower() +
#     ' | ' + df_tmp['atividade_secundaria1'].str.replace(".*outr[ao]s.*n[aã]o especif.*", '', regex=True).str.lower() +
#     ' | ' + df_tmp['atividade_secundaria2'].str.replace(".*outr[ao]s.*n[aã]o especif.*", '', regex=True).str.lower() +
#     ' '
# )

# normalizacao
df_tmp.replace({
    'atividade_caracteristicas': {
        r'á': 'a', r'à': 'a', r'â': 'a', r'ã': 'a', r'ä': 'a',
        r'é': 'e', r'è': 'e', r'ê': 'e', r'ẽ': 'e', r'ë': 'e',
        r'í': 'i', r'ì': 'i', r'î': 'i', r'ĩ': 'i', r'ï': 'i',
        r'ó': 'o', r'ò': 'o', r'ô': 'o', r'õ': 'o', r'ö': 'o',
        r'ú': 'u', r'ù': 'u', r'û': 'u', r'ũ': 'u', r'ü': 'u',
        r'ç': 'c',
    }
}, regex=True, inplace=True)

def replace_multiple(df_tmp, column, dict_replaces):
    i = 0
    for key, value in dict_replaces.items():
        i += 1
        if (i % 5) == 0:
            print("step:", i, "/", len(dict_replaces.keys()))
        df_tmp.replace({column: {k: key for k in value} }, regex=True, inplace=True)
    return df_tmp

# tirar stop words
df_tmp = replace_multiple(df_tmp, 'atividade_caracteristicas', {
    '': [r'\-'],
    ' ': [r' e ', r' de ', r' da ', r' do ', r' em ', r' com ', r'\,', r'\.', r'p\/', r' para ', r' por ', r' a ', r' e\/ou ', r' no ', r' na '],
})
# df_tmp.replace({
#     'atividade_principal': {
#         r'\-': '', r' e ': ' ', r' de ': ' ', r' da ': ' ', r' do ': ' ', r' em ': ' ', r' com ': ' ',
#         r'\,': ' ', r'\.': ' ', r'p\/': ' '
#     }
# }, regex=True, inplace=True)

# corrigir texto basico
df_tmp = replace_multiple(df_tmp, 'atividade_caracteristicas', {
    ' acougue ': [r' a ougue '],
    ' fabrica ': [r' fabrica ao '],
    ' refeicoes ': [r' refei oes '],
    #
    ' livraria papelaria comercio ': [r' liv pap com '],
    ' locacao administracao imovel ': [r' loc adm imoveis '],
    #
    ' artigo couro ': [r' r co '],
    ' aparelho eletronico ': [r' apar elet ', r' aparelho elet ', r' apar eletronico '],
    ' artigo religioso ': [r' art relig '],
    ' arte religiosa ': [r' arte relig ', r' arte rel '],
    ' disco fita ': [r' disc fit ', r' disco fit ', r' disc fita '],
    ' jogo eletronico ': [r' jogos ele ', r' jogos eletronicos '],
    ' livraria papelaria ': [r' livr papel ', r' livrar papel ', r' livraria papel ', r' livraria pap '],
    ' livraria comercio ': [r' livr com '],
    ' livro revista ': [r' livros e revi ', r' livros revi ', r' livro e revi ', r' livro revi '],
    ' produto natural ': [r' prod nat '],
    ' refeicao embalada ': [r' refei emb '],
    ' representante comercial ': [r' repres comerciais ', r' represent comerc ', r' represent comerciais ', r' representantes comercial ', r' representantes comerciais '],
    ' comercio varejo ': [r' lj com var ', r' lj cm vr ', r' lj varejo ', r' com varejo '],
    ' comercio atacado ': [r' lj com atac ', r' lj com a '],
    ' locadora cd ': [r' loc cd '],
    ' perfume cosmetico ': [r' perfume cos '],
    ' roupa malha ': [r' roupas mal ', r' roupa mal '],
    ' doce salgado ': [r' doces sal '],
    ' reparacao eletrica ': [r' reparacao ele ', r' repar ele ', r' rep ele '],
    ' equipamento eletrico ': [r' equip elet '],
    ' processamento dado ': [r' process dad '],
    #
    ' acessorio ': [r' ace ', r' aces ', r' acess ', r' acessorios '],
    ' alimento ': [r' alim ', r' prod ali ', r' alimentos ', r' alimentoe ', r' prod alim ', r' produto alimentar ', r' alimentici ', r' alimenticios '],
    ' conserva ': [r' conservas '],
    ' carne ': [r' carn ', r' carnes '],
    ' armacao ': [r' armar ', r' armacoes '],
    ' arte ': [r' artes ', r' art '],
    ' manufaturado ': [r' manuf '],
    ' textil ': [r' texteis '],
    ' concurso ': [r' concursos '],
    ' direito ': [r' direitos '],
    ' sanitario ': [r' sanitaria ', r' sanitarias ', r' sanitarios '],
    ' praga ': [r' pragas '],
    ' artefato ': [r' artef ', r' arteff ', r' artefatos '],
    ' artesanato ': [r' artesan ', r' artesanatos ', r' artesanat '],
    ' artesanal ': [r' artesanais '],
    ' artigo ': [r' artg ', r' arti ', r' artig ', r' artigos '],
    ' motoneta ': [r' motonetas '],
    ' atendendimento ': [r' atend '],
    ' atendimento ': [r' atend ', r' c\/atendimento '],
    ' bijuteria ': [r' bi ', r' bij ', r' biju ', r' bijut ', r' bijute ',  r'bijuter', r' bijuteri ', r' bijuterias '],
    ' bolsa ': [r' bols ', r' bolsas '],
    ' brinquedo ': [r' brin ', r' brinqued ', r' brinq ', r' brinquedos '],
    ' calcado ': [r' cal ', r' calc ', r' calcad ', r' calcados '],
    ' obra ': [r' obras '],
    ' telecomunicacao ': [r' telecomunicacoes '],
    ' paciente ': [r' pacientes '],
    ' seguranca ': [r' seguran '],
    ' seguro ': [r' seguros '],
    ' carga ': [r' cargas '],
    ' doce ': [r' doces '],
    ' pessoa ': [r' pessoas '],
    ' ceramica ': [r' ceram ', r' ceramicas '],
    ' animal ': [r' animais '],
    ' vivo ': [r' vivos '],
    ' optico ': [r' opticos '],
    ' tabaco ': [r' tabac '],
    ' cama ': [r' cam '],
    ' chocolate ': [r' chocolates '],
    ' comercio ': [r' comerc ', r' comer '],
    ' atacado ': [r' atacadista '],
    ' comunicao ': [r' comunic '],
    ' confeitaria ': [r' confei ', r' confeit ', r' confeita ', r' confeitar ', r' confeitari '],
    ' cosmetico ': [r' cosm ', r' cosmet ', r' cosmeticos '],
    ' couro ': [r' cour ', r' couros '],
    ' embalad ': [r' embalada '],
    ' embalado ': [r' emba '],
    ' embalado ': [r' embalados '],
    ' escritorio ': [r' escr ', r' escrt ', r' escri ', r' escrit ', r' escritorios '],
    ' fita cassete ': [r' fit casset'],
    ' frios ': [r' frio '],
    ' jornal ': [r' jorn ', r' jornais '],
    ' juteria ': [r' j ', r' ju ', r' juter '],
    ' lanchonete ': [r' lanchon ', r' lanch ', r' lanchonet ', r' lanchonetes '],
    ' livraria ': [r' livr', r'livrar '],
    ' livro ': [r' livros '],
    ' maquiagem ': [r' maq '],
    ' material ': [r' mat ', r' mate ', r' mater ', r' materi ', r' materia ', r' materiais '],
    ' mercearia ': [r' merc ', r' merce ', r' mercea ', r' mercear ',],
    ' mesa ': [r' mes '],
    ' metal ': [r' met ', r' metais '],
    ' moldura ': [r' mold ', r' molduras '],
    ' natural ': [r' naturais '],
    ' otico ': [r' otic ', r' otica ', r' oticas ', r' oticos '],
    ' panificadora ': [r' panif '],
    ' pao ': [r' paes '],
    ' pastelaria ': [r' pastelar '],
    ' perfume ': [r' perfum ', r' perf ', r' perfumes '],
    ' pizzaria ': [r' pizzar ', r' pizzari '],
    ' plastico ': [r' plast '],
    ' plano ': [r' planos '],
    ' quadro ': [r' quadr ', r' quadros '],
    ' rapida ': [r' rapidas '],
    ' refeicao ': [r' refei ', r' refeic ', r' refeicoes '],
    ' restaurante ': [r' rest ', r' restaur '],
    ' revista ': [r' rev ', r' revist '],
    ' salgado ': [r' salg '],
    ' salgadinho ': [r' salgadinhos '],
    ' servico ': [r' serv ', r' servicos '],
    ' social ': [r' socia ', r' sociais '],
    ' sorvete ': [r' sorvetes ', r' sorv '],
    ' sorveteria ': [r' sorveter '],
    ' suco ': [r' sucos '],
    ' peca ': [r' pecas '],
    ' publicacao ': [r' publicacoes '],
    ' camioneta ': [r' camionetas '],
    ' automovel ': [r' automoveis '],
    ' siderurgico ': [r' siderurgicos '],
    ' residuo ': [r' residuos '],
    ' perigoso ': [r' perigosos '],
    ' religioso ': [r' religiosa ', r' religiosas ', r' religiosos '],
    ' filosofia ': [r' filosofica ', r' filosofico ', r' filosoficas ', r' filosoficos '],
    ' tapecaria ': [r' tapecar ', r' tapecarias '],
    ' tecido ': [r' tec ', r' tecidos '],
    ' importado ': [r' imp ', r' import '],
    ' cinto ': [r' cintos '],
    ' associativa ': [r' associativas '],
    ' patronal ': [r' patronais '],
    ' musical ': [r' musicais '],
    ' urgencia ': [r' urgencias '],
    ' joia ': [r' joias '],
    ' objeto ': [r' objetos '],
    ' fotografia ': [r' fotografias '],
    ' aerea ': [r' aereas '],
    ' banco ': [r' bancos '],
    ' cartao ': [r' cartoes '],
    ' profissional ': [r' profissionais '],
    ' relogio ': [r' relogios '],
    ' consorcio ': [r' consorcios '],
    ' lubrificante ': [r' lubrificantes '],
    ' placa ': [r' placas '],
    ' bovino ': [r' bovinos '],
    ' fechada ': [r' fech ', r' fecha ', r' fechad ', r' fechadas '],
    ' veterinario ': [r' veterinaria ', r' veterinarias ', r' veterinarios '],
    ' varejo ': [r' vr ', r' varj ', r' var ', r' vare ', r' varej ', r' vrj ', r' varejista '],
    ' veiculo ': [r' vei ', r' veic ', r' veicu ', r' veicul ', r' veiculos '],
    ' vestuario ': [r' vest ', r' vestu ', r' vestua ', r' vestuar ', r' vestuari ', r' vestuarios '],
    ' escola ': [r' esc ', r' esco ', r' escol '],
    ' disco ': [r' disc ', r' dis ', r' discos '],
    ' produto ': [r' p ', r' produtos ', r' prod '],
    ' refrigerante ': [r' refrig '],
    ' lanche ': [r' lanches '],
    ' antiguidade ': [r' antiguidades '],
    ' gema ': [r' gemas '],
    ' documento ': [r' doc ', r' documentos '],
    ' esporte ': [r' esport ', r' esportes ', r' esportivas ', r' esportiva ', r' esportivo ', r' esportivos '],
    ' lavanderia ': [r' lav '],
    ' instituto ': [r' inst ', r' insti ', r' instit ', r' institu ', r' instituicao ', r' instituicoes '],
    ' educacao ': [r' educ '],
    ' assistencia ': [r' assist '],
    ' clinica ': [r' clin '],
    ' cabeleireiro ': [r' cabelei ', r' cabeleireiros '],
    ' depilacao ': [r' depilaca '],
    ' edificios ': [r' edificio '],
    ' agencia ': [r' agencias '],
    ' agente ': [r' agentes '],
    ' viagem ': [r' viagens '],
    ' sociedade ': [r' sociedades '],
    ' combinado ': [r' combinados '],
    ' proprio ': [r' proprios '],
    ' recurso ': [r' recursos '],
    ' fabrica ': [r' fab ', r' fabricacao ', r' fabric '],
    ' roupa ': [r' roupas ', r' roup '],
    ' solda ': [r' soldas '],
    ' hortifruit ': [r' hortifr '],
    ' adubo ': [r' adubos '],
    ' fertilizante ': [r' fertilizantes '],
    ' organicomineral ': [r' organominera '],
    ' estrutura ': [r' estruturas '],
    ' metalica ': [r' metalicas '],
    ' instrumento ': [r' instrumentos '],
    ' naoeletronico ': [r' naoeletronicos '],
    ' utensilio ': [r' utensilios '],
    ' maquina ': [r' maquinas '],
    ' aparelho ': [r' apar ', r' apare ', r' aparel ', r' aparelh ', r' aparelhos '],
    ' oleo ': [r' oleos '],
    ' vegetal ': [r' vegetais '],
    ' refinado ': [r' refinados '],
    ' periferico ': [r' perifericos '],
    ' equipamento ': [r' equip ', r' equipamentos '],
    ' loja ': [r' lj ', r' loj ', r' lojas '],
    ' armarinho ': [r' armarinhos '],
    ' domestico ': [r' domest ', r' domestica ', r' domesticas ', r' domesticos '],
    ' ornamento ': [r' orn ', r' orna ', r' ornam ', r' orname '],
    ' aquario ': [r' aquarios '],
    ' embalagem ': [r' embalagens '],
    ' hidraulico ': [r' hidraulica ', r' hidraulicas ', r' hidraulicos '],
    ' granito ': [r' granitos '],
    ' instalacao ': [r' instalacoes '],
    ' reparacao ': [r' rep '],
    ' vassoura ': [r' vassouras '],
    ' espanador ': [r' espanadore ', r' espanadores '],
    ' industria ': [r' ind '],
    ' industrial ': [r' indus ', r' industriais '],
    ' propaganda ': [r' propag '],
    ' imovel ': [r' imoveis ', r' moveis '],
    ' imobiliario ': [r' imobiliarios '],
    ' combustivel ': [r' combustiveis '],
    ' gasoso ': [r' gasosos '],
    ' rede ': [r' redes '],
    ' urbana ': [r' urbanas '],
    ' sistema ': [r' sistemas '],
    ' central ': [r' centrais '],
    ' diario ': [r' diarios '],
    ' discoteca ': [r' discotecas '],
    ' danceteria ': [r' danceterias '],
    ' salao ': [r' saloes '],
    ' administracao ': [r' admin ', r' admini ', r' adminis ', r' administ ', r' administr ', r' administrativas ', r' administrativa ', r' administrativo ', r' administrativos '],
    ' desenvolvimento ': [r' desenvo ', r' desenvol ', r' desenvolv ', r' desenvolvi ', r' desenvolvim '],
    ' empresa ': [r' empr ', r' empresas '],
    ' unidade ': [r' unidades '],
    ' intima ': [r' intimas '],
    ' barragem ': [r' barragens '],
    ' recreativa ': [r' recreativas ', r' recreativos '],
    ' deposito ': [r' depositos '],
    ' mercadoria ': [r' mercadorias '],
    ' terceiro ': [r' terceiros '],
    ' armazem ': [r' armazens '],
    ' geral ': [r' gerais '],
    ' guardamovel ': [r' guardamoveis '],
    ' correspondente ': [r' correspondentes '],
    ' financeira ': [r' financeiras '],
    ' corretora ': [r' corretoras ', r' corretores '],
    ' titulo ': [r' titulos '],
    ' valor ': [r' valores '],
    ' mobiliario ': [r' mobiliarios '],
    ' ovino ': [r' ovinos '],
    ' suino ': [r' suinos '],
    ' curso ': [r' cursos '],
    ' comercial ': [r' comerciais '],
    ' representante ': [r' repr ', r' repre ', r' repres ', r' reppres '],
    ' holding ': [r' holdings '],
    ' medicamento ': [r' medicam '],
    ' floricultura ': [r' floricult '],
    ' peixe ': [r' peixes '],
    ' especial ': [r' especiais '],
    ' papel ': [r' pape '],
    ' pasta ': [r' pastas '],
    ' estabelecimento ': [r' estab '],
    ' ensino ': [r' ens '],
    ' idioma ': [r' idio ', r' idiom ', r' idiomas '],
    ' estacionamento ': [r' estacion ', r' estaciona ', r' estacionam ', r' estacioname ', r' estacionamen ', r' estacionament ',],
    ' lavagem ': [r' lavag ',],
    ' adesivo ': [r' adesivos ',],
    ' selante ': [r' selantes ',],
    ' prato ': [r' pratos ',],
    ' pronto ': [r' prontos ',],
    ' didatico ': [r' didaticos ',],
    ' congelado ': [r' congelados ',],
    ' festa ': [r' fest ', r' festas '],
    ' quimico ': [r' quim ', r' quimi ', r' quimic ', r' quimicos ',],
    ' trabalho ': [r' trabalhos ',],
    ' grafico ': [r' graficos ',],
    ' eletronico ': [r' eletr ', r' eletro ', r' eletron ', r' eletroni ', r' eletronic ', r' eletronicos '],
    ' bebida ': [r' beb ', r' bebi ', r' bebid ', r' bebidas '],
    ' decoracao ': [r' decorac '],
    ' estacao ': [r' estacoes '],
    ' luminaria ': [r' luminarias '],
    ' processamento ': [r' process '],
    ' queijo ': [r' queij '],
    ' frio ': [r' frios '],
    ' bar ': [r' bares '],
    ' feira ': [r' feiras '],
    ' cadastral ': [r' cadastrais '],
    ' informacao ': [r' informacoes '],
    ' cobranca ': [r' cobrancas '],
    ' especializacao ': [r' especializados '],
    ' casa ': [r' casas '],
    ' suprimento ': [r' suprimentos '],
    ' suvenil ': [r' suvenires '],
    ' exposicao ': [r' exposicoes '],
    ' loterica ': [r' lotericas '],
    ' embarcacao ': [r' embarcacoes '],
    ' infantil ': [r' infantis '],
    ' condominio ': [r' condominios '],
    ' predio ': [r' predial ', r' prediais '],
    ' corretivo ': [r' corretivos '],
    ' agricola ': [r' agricolas '],
    ' defensivo ': [r' defensivos '],
    ' dado ': [r' dados '],
    ' atividade ': [r' atividades '],
    ' variedade ': [r' variedades '],
    ' assessoria ': [r' assess '],
    ' pneumatico ': [r' pneumaticos '],
    ' metalurgico ': [r' metalurgicos '],
    ' ortopedia ': [r' ortop ', r' ortopedico '],
    ' sindical ': [r' sindicais '],
    ' organizacao ': [r' organizacoes '],
    ' advocaticio ': [r' advocaticios '],
    ' local ': [r' locais '],
    ' farmaceutico ': [r' farmaceuticos '],
    ' formula ': [r' formulas '],
    ' customizavel ': [r' customizaveis '],
    ' eletrodomestico ': [r' eletrodomesticos '],
    ' negocio ': [r' negocios '],
    ' venda ': [r' vendas '],
})

df_tmp.replace({
    'atividade_caracteristicas': {
        r'\s+': ' ',
    }
}, regex=True, inplace=True)

phrases_basura = [
    r' material semelhante ',
    r' bolsa semelhante qualquer material ',
    r' outros produto alimenticio ',
    r' nao esp ',
    r' similar ', r' similares ',
    r' outros trabalho ',
    r' inclusive ',
    r' outros artefato ',
    r' nao especificados anteriormente ',
    r' diversa ', r' diversas ', r' diverso ', r' diversos ',
    r' outros produto grafico ',
    r' outras atividades ',
    r' relacionados ',
    r' uso ',
    r' demais derivados petroleo ',
    r' outros equipamento naoeletronico ',
    r' outros veiculo recreativos ',
    r' nao especificadas anteriormente ',
    r' \(sem consumo no local\) ',
    r' \(servicos valet\) ',
    r' outros servico cuidados beleza ',
    r' exceto holding ',
    r' outros artigo domestico ',
    r' outros servico $',
    r' outros servico t $',
    r' outros estabelecimentos? especializados? servir bebida $',
    r' outros servico informacao internet $',
    r' outros servico tecnologia informacao $',
    r' outros servico cuidad $',
    r' outras estrutura temporario exceto andaimes? $',
    r' outros veiculo recreativos\; ',
    r' outros estabelecimentos? especializados? $',
    r' outros exames? analogos $',
    r' outros equipamento artigo $',
    r' outras publicacao $',
    r' outras maquina equipamento comercia $',
    r' outros servico turismo $',
    r' outros produt $',
    r' outros produto g $',
    r' outros produto nao especifi $',
    r' ou especializado produto alimento $',
    r' exceto imobiliarios? $',
    r' exceto consultoria tecnica especifica $',
    r' exceto encadernacao plastificacao $',
    r' exceto imovel $',
    r' exceto roupa int $',
    r' exceto prontosocorro unidade atendimento urgencia $',
    r' exceto produto perigoso mudancas? intermunicipal interestadual internacional $',
    r' exceto o transporte maritimo $',
    r' exceto roupa intima $',
    r' exceto tomografia $',
    r' exceto aere $',
    r' exceto roupa intima as confeccionadas sob medida $',
    r' exceto luminosos? $',
    r' exceto aerea submarina $',
    r' exceto ressonancia magnetica $',
    r' exceto informatica comunicacao $',
    r' exceto profissional seguranca $',
    r' exceto veiculo comunicacao $',
    r' exceto condominio predio $',
    r' exceto os servico imovel atendimento urgencia $',
    r' exceto armazem geral guardamovel $',
    r' exceto confeccao $',
    r' exceto oleo milho $',
    r' exceto seguranca protecao $',
    r' exceto produto perigoso mudancas municipal $',
    r' exceto construcao $',
    r' exceto andaimes? $',
    r' exceto lubrificante nao realizado transportador retalhista \(trr\) $',
    r' exceto loja departamentos? $',
    r' exceto organ $',
    r' exceto pront $',
    r' exceto $',
    r' exceto produto pe $',
]

for phrase in phrases_basura:
    df_tmp = replace_multiple(df_tmp, 'atividade_caracteristicas', { ' ': [phrase,] })
        
# corrigir espacos
df_tmp.replace({
    'atividade_caracteristicas': {
        r'\s+': ' ',
    }
}, regex=True, inplace=True)

df_tmp['atividade_caracteristicas'] = df_tmp['atividade_caracteristicas'].str.strip()
'ok'
def reorder_words(phrase):
    arr = phrase.strip().split(' ')
    arr.sort()
    return ' '.join(arr)

df_tmp['atividade_caracteristicas'] = df_tmp['atividade_caracteristicas'].apply(reorder_words)
df_original = df_tmp
del df_tmp
(
    df_original[['atividade_caracteristicas', 'atividade_principal', 'atividade_secundaria1', 'atividade_secundaria2', 'atividade_principal_agregada']]
        .groupby(['atividade_caracteristicas', 'atividade_principal', 'atividade_secundaria1', 'atividade_secundaria2', 'atividade_principal_agregada'])
        .size().reset_index(name='counts')
    .to_csv('atividade_caracteristicas.csv')
)
# df = df_original
df = df_original[df_original['bairro'] == 'BATEL']
re_map = {
    'ABATEDOURO': 'Slaughterhouse',
    'ACADEMIA': 'GYM',
    'ACOUGUE': 'Butcher Shop',
    'ADMINISTRADORA': 'Administrator',
    'AGENCIA': 'Agency',
    'ALBERGUE': 'Hostel',
    'ALUGUEL': 'Rent',
    'ARMAZENS': 'Warehouse',
    'ASSOCIACAO': 'Association',
    'ATELIER': 'Atelier',
    'AUTO_ESCOLA': 'Driving School',
    'BANCA': 'Cards Bank',
    'BANCO': 'Bank',
    'BANCOS': 'Bank',
    'BAR': 'Bar',
    'BORRACHARIA': 'Tire House',
    'BOX': 'Box',
    'CABELEIREIRO': 'HairDresser',
    'CARTORIO': 'Notary\'s Office',
    'CHAVEIRO': 'Locksmith',
    'CLUBE': 'Club',
    'COM_ATACADISTA': 'Wholesale Trade',
    'COM_VAREJO': 'Retail Trade',
    'CONSTRUTORA': 'Construction',
    'COOPERATIVA': 'Co-op',
    'CORRETORA': 'Broker',
    'CRIADOUROS': 'Breeding Grounds',
    'CULTIVO_GERAL': 'General Farming',
    'CULTURA': 'Culture',
    'CURSOS_GERAL': 'Gereral Courses',
    'DEPOSITO': 'Deposit',
    'DISTRIBUIDORA': 'Distributor Company',
    'EDITORAS': 'Editors',
    'EDUCACAO': 'Education',
    'ENTRETENIMENTO': 'Entertainment',
    'ESCRITORIO': 'Office',
    'ESTACIONAMENTO': 'Parking lot',
    'ESTRACAO_BENS_NATURAIS': 'Extraction of natural resources',
    'FABRICA': 'Factory',
    'FARMACIA': 'Drugstore',
    'FLORICULTURA': 'Floriculture',
    'GALERIA_ARTE': 'Art Gallery',
    'HOTEL': 'Hotel',
    'INDUSTRIA': 'Industry',
    'LABORATORIO': 'Laboratory',
    'LANCHONETE': 'Snack Bar',
    'LAVANDERIA': 'Laundry',
    'LIVRARIA': 'Bookstore',
    'LOJA': 'Store',
    'LOTERICA': 'Lottery House',
    'MANUFATURA': 'Manufacture Services',
    'MANUTENCAO_REPAROS': 'Repair Services',
    'MERCEARIA': 'Grocery Store',
    'ODONTOLOGIA': 'Dentistry',
    'OFICINA_GERAL': 'General Manufactury',
    'ORFANATO': 'Orphanage',
    'PANIFICADORA': 'Bakery',
    'PAPELARIA': 'Stationary',
    'PASTELARIA': 'Pastry Shop',
    'PEIXARIA': 'Fish Shop',
    'PENSAO': 'Inn',
    'POSTO_COMBUSTIVEL': 'Fuel Station',
    'RELOGOARIA': 'Watch House',
    'RESTAURANTE': 'Restaurant',
    'SALAO_BELEZA': 'Beauty Shop',
    'SERV_ALIMENTACAO': 'Food Services',
    'SERV_GERAL': 'Gereral Services',
    'SERV_HOSPITALAR': 'Hospital Services',
    'TERMINAL_RODOVIARIO': 'Bus Terminal',
    'TRANSPORTADORA': 'Shipping Company',
    'TRANSPORTE_GERAL': 'Gereral Transportation',
}
df['atividade_principal_agregada'] = df['atividade_principal_agregada'].map(re_map.get)
df
_tmp_1 = df[['numero_do_alvara']].copy()

for axis in ('latitude', 'longitude'):
    coord_min = df[axis].min()
    coord_max = df[axis].max()
    if (coord_max - coord_min) == 0:
        _tmp_1.loc[:, 'normalized_' + axis] = 0
    else:
        _tmp_1.loc[:, 'normalized_' + axis] = ((df[axis] - coord_min) / (coord_max - coord_min))

_tmp_1['normalized_latitude'] = _tmp_1['normalized_latitude'] * 1
_tmp_1['normalized_longitude'] = _tmp_1['normalized_longitude'] * 1

df['data_inicio_atividade__year'] = pd.to_datetime(df['data_inicio_atividade']).dt.year
years = df['data_inicio_atividade__year']
years_min = years.min()
years_max = years.max()
_tmp_1['normalized_year'] = (years - years_min) / (years_max - years_min)
_tmp_1['normalized_year'] = _tmp_1['normalized_year'] * 1

activies = np.unique(df['atividade_caracteristicas'])
for activity in activies:
    _tmp_1['normalized_' + activity] = 0
    _tmp_1.loc[df['atividade_caracteristicas'] == activity, 'normalized_' + activity] = 1 * 1

    # activies = np.unique(df['atividade_principal_agregada'])
    # for activity in activies:
    #     _tmp_1['normalized_' + activity] = 0
    #     _tmp_1.loc[df['atividade_principal_agregada'] == activity, 'normalized_' + activity] = 1 * 1
    
# for col in ['normalized_latitude', 'normalized_longitude', 'normalized_year']:
#     _tmp_1[col] = _tmp_1[col].astype(np.float32)
# for activity in activies:
#     _tmp_1['normalized_' + activity] = _tmp_1['normalized_' + activity].astype(np.uint8)

normalized_df = {
    'space_time_type': _tmp_1.copy()
}

normalized_df['space_type'] = _tmp_1.copy()
normalized_df['space_type']['normalized_year'] = normalized_df['space_type']['normalized_year'] * 0

normalized_df['time_type'] = _tmp_1.copy()
normalized_df['time_type']['normalized_latitude'] = normalized_df['time_type']['normalized_latitude'] * 0
normalized_df['time_type']['normalized_longitude'] = normalized_df['time_type']['normalized_longitude'] * 0

normalized_df['space_time'] = _tmp_1.copy()
for activity in activies:
    normalized_df['space_time']['normalized_' + activity] = 0

normalized_df['space'] = _tmp_1.copy()
normalized_df['space']['normalized_year'] = normalized_df['space']['normalized_year'] * 0
for activity in activies:
    normalized_df['space']['normalized_' + activity] = 0
    
normalized_df['time'] = _tmp_1.copy()
normalized_df['time']['normalized_latitude'] = normalized_df['time']['normalized_latitude'] * 0
normalized_df['time']['normalized_longitude'] = normalized_df['time']['normalized_longitude'] * 0
for activity in activies:
    normalized_df['time']['normalized_' + activity] = 0

normalized_df['type'] = _tmp_1.copy()
normalized_df['type']['normalized_year'] = normalized_df['type']['normalized_year'] * 0
normalized_df['type']['normalized_latitude'] = normalized_df['type']['normalized_latitude'] * 0
normalized_df['type']['normalized_longitude'] = normalized_df['type']['normalized_longitude'] * 0

# normalized_df['street_time_type'] = normalized_df['time_type'].copy()
# enderecos = np.unique(df['endereco'])
# for endereco in enderecos:
#     normalized_df['street_time_type']['normalized_' + endereco] = 0
#     normalized_df['street_time_type'].loc[df['endereco'] == endereco, 'normalized_' + endereco] = 1 * 1

# normalized_df['street_time_type'] = normalized_df['time_type'].copy()
# endereco_labels, endereco_levels = pd.factorize(df['endereco'])
# normalized_df['street_time_type']['street'] = endereco_labels

# normalized_df['street_time_type']

normalized_df['space_time_type']
K = range(1,30)
distortions = {}
# types_clusters = ['street_time_type', 'space_time_type', 'space_type', 'time_type', 'space_time', 'time', 'space', 'type']
types_clusters = normalized_df.keys()
for type_cluster in types_clusters:
    print('type_cluster:', type_cluster)
    _tmp_2 = normalized_df[type_cluster]
    X = _tmp_2.loc[:, _tmp_2.columns != 'numero_do_alvara'].values

    distortions[type_cluster] = []
    for k in K:
        if k % 5 == 0:
            print("\tk:", k)
#         print("\tk:", k)
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions[type_cluster].append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

'ok'
# %matplotlib notebook
# %matplotlib notebook
%matplotlib inline
# Plot the elbow
plt.xlabel('K (number of clusters)')
plt.ylabel('Distortion (less is better)')
plt.title('Elbow Method')
for type_cluster in ['space_time_type', 'space_time', 'space_type', 'time_type']:
    plt.plot(K, distortions[type_cluster])
for type_cluster in ['space', 'time', 'type']:
    plt.plot(K, distortions[type_cluster], '--')
plt.legend(['space × time × type', 'space × time', 'space × type', 'time × type', 'space only', 'time only', 'type only'], loc='upper right')
plt.axvline(x=5, color='k', linestyle='--', alpha=0.5)
# plt.xticks([0, 3, 5, 10, 15, 20, 25, 30])
# plt.xticks(arange(30, step=3))
plt.savefig('elbow-method.png',bbox_inches='tight')
plt.show()
kmeans = {}
# for type_cluster in {'space_time_type'}:
# for type_cluster in {'street_time_type'}:
for type_cluster in {'space_time_type', 'space_type', 'time_type', 'space_time'}:
    print("type_cluster:", type_cluster)
    _tmp_2 = normalized_df[type_cluster]
    X = _tmp_2.loc[:, _tmp_2.columns != 'numero_do_alvara'].values
    kmeans[type_cluster] = {}
    for num_clusters in range(3, 30):
        if num_clusters % 5 == 0:
            print('\tnum_clusters:', num_clusters)
        kmeans[type_cluster][num_clusters] = KMeans(n_clusters=num_clusters).fit(X)
'OK'
%matplotlib inline
# %matplotlib notebook
# %matplotlib notebook

create_dir_if_doesnt_exists('scatter--tempo-vs-tipo')

# types_cluster = ['space_time_type']
types_cluster = {'space_time_type', 'space_type', 'time_type', 'space_time'}
for type_cluster in types_cluster:
    create_dir_if_doesnt_exists('scatter--tempo-vs-tipo/{}'.format(type_cluster))

distortions = {} # free distortions
for type_cluster in types_cluster:
#     for k in [5]:
    for k in range(3, 30):
#     for k in range(3, 5):
        _tmp_2 = normalized_df[type_cluster].copy()
        labels = kmeans[type_cluster][k].labels_
        _tmp_2['label'] = labels

        normalized_df_merge_df = pd.merge(df, _tmp_2, on='numero_do_alvara').copy()
        
        clusters_informations = {}
        for cluster_k in range(0, k):
            cluster = normalized_df_merge_df[normalized_df_merge_df['label'] == cluster_k].copy()
            if type_cluster == 'space_time':
                for activity in activies:
                    cluster['normalized_' + activity] = df['atividade_caracteristicas'] == activity
                    # cluster['normalized_' + activity] = df['atividade_principal_agregada'] == activity
                
                
            _tmp_3 = pd.DataFrame({'y': activies, 'x': [cluster['normalized_' + activity].sum() for activity in activies]})
            _tmp_3 = _tmp_3[_tmp_3['x'] != 0]
            clusters_informations[cluster_k] = {
                'label': cluster_k,
                'total': _tmp_3.sum()['x'],
                'total_types': _tmp_3.shape[0],
                'activities': _tmp_3.sort_values(['x'], ascending=[False])['y'].values,
            }
            cluster_i = clusters_informations[cluster_k]
            complete_legend = "[{} .. {}]".format(cluster['data_inicio_atividade'].min(), cluster['data_inicio_atividade'].max())
            simple_legend = ','.join(cluster_i['activities'])
            MAX_LENGTH = 60#20
            if len(simple_legend) > MAX_LENGTH:
                simple_legend = simple_legend[:MAX_LENGTH] + '… ({})'.format(cluster_i['total_types'])
            complete_legend = complete_legend + ' ' + simple_legend
            complete_legend = complete_legend + " #{}".format(cluster_i['total'])
            cluster_i['complete_legend'] = complete_legend
            cluster_i['simple_legend'] = simple_legend
        clusters_informations = pd.DataFrame(data=clusters_informations).T
        clusters_informations = clusters_informations.sort_values(['total_types', 'total'], ascending=[True, False])
        #
        i = 0
        normalized_df_merge_df['number_activity'] = i
        _last_activities = set()
        for cluster_k, cluster in clusters_informations.iterrows():
            _tmp_activies = cluster['activities']
            for activity in _tmp_activies:
                if not (activity in _last_activities):
                    i += 1
                    normalized_df_merge_df.loc[normalized_df_merge_df['atividade_caracteristicas'] == activity, 'number_activity'] = i
                    # normalized_df_merge_df.loc[normalized_df_merge_df['atividade_principal_agregada'] == activity, 'number_activity'] = i
                    _last_activities.add(activity)

        groups = normalized_df_merge_df.groupby('label')

        fig, ax = plt.subplots()#(figsize=(4, 6))
        ax.margins(0.05)
        plt.gca().invert_yaxis()
        for cluster_k, cluster in clusters_informations.iterrows():
            ax.plot(groups.get_group(cluster_k).data_inicio_atividade__year, \
                    groups.get_group(cluster_k).number_activity, marker='o', \
                    #linestyle='', alpha=0.005, ms=5, label=cluster['simple_legend'])#curitiba
                    linestyle='', alpha=0.05, ms=5, label=cluster['complete_legend'])#curitiba

#         leg = ax.legend(framealpha=1, title="Clusters", loc='lower left')
        leg = ax.legend(framealpha=1, title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))
        for lh in leg.legendHandles: 
            lh._legmarker.set_alpha(1)

        ax.set(ylabel='Type (ordered by total amount)', xlabel='Creation Date')
        ax.set_yticklabels([])
        filename = "scatter-{}---{}.png".format(type_cluster, k)
        print("filename:", filename)
#         filename = "scatter--tempo-vs-tipo/{}/scatter--tempo-vs-tipo--{}.png".format(type_cluster, k)
        plt.savefig(filename,bbox_inches='tight')
        plt.close('all')

# ax
'ok'
# clusters_informations
from matplotlib import colors

%matplotlib inline
#%matplotlib notebook
#%matplotlib notebook

def run_plot(used_attribute='atividade_principal_agregada'):
    # for type_cluster in ['street_time_type']:
    for type_cluster in ['space_time_type']:
    # for type_cluster in {'space_time_type', 'space_type', 'time_type', 'space_time'}:
        for k in [5]:
    #     for k in range(3, 30):
    #     for k in range(3, 5):
            _tmp_2 = normalized_df[type_cluster].copy()
            labels = kmeans[type_cluster][k].labels_
            _tmp_2['label'] = labels

            normalized_df_merge_df = pd.merge(df, _tmp_2, on='numero_do_alvara').copy()

            table_df = normalized_df_merge_df \
                .groupby([used_attribute, 'data_inicio_atividade__year']).size().reset_index(name='counts') \
                .pivot(used_attribute, "data_inicio_atividade__year", "counts") \
                .fillna(0).astype(np.int32)

            clusters_informations = {}
            for cluster_k in range(0, k):
                cluster = normalized_df_merge_df[normalized_df_merge_df['label'] == cluster_k].copy()
                if type_cluster == 'space_time':
                    for activity in activies:
                        cluster['normalized_' + activity] = df[used_attribute] == activity


                _tmp_3 = pd.DataFrame({'y': activies, 'x': [cluster['normalized_' + activity].sum() for activity in activies]})
                _tmp_3 = _tmp_3[_tmp_3['x'] != 0]
                clusters_informations[cluster_k] = {
                    'label': cluster_k,
                    'total': _tmp_3.sum()['x'],
                    'total_types': _tmp_3.shape[0],
                    'activities': _tmp_3.sort_values(['x'], ascending=[False])['y'].values,
                }
            clusters_informations = pd.DataFrame(data=clusters_informations).T
            clusters_informations = clusters_informations.sort_values(['total'], ascending=[True])
#             clusters_informations = clusters_informations.sort_values(['total_types', 'total'], ascending=[True, False])

            list_activities = []
            for cluster_k, cluster in clusters_informations.iterrows():
                _tmp_activies = cluster['activities']
                for activity in _tmp_activies:
                    if not (activity in list_activities):
                        list_activities.append(activity)

            ########## table_df = table_df.reindex_axis(list_activities)
            table_df = table_df.reindex_axis(
                normalized_df_merge_df.groupby(used_attribute).size().sort_values(ascending=False).keys()
            )

            plt.tight_layout()
            f, ax = plt.subplots(figsize=(10, 150))
            sns.heatmap(table_df, annot=False, fmt="d", #linewidths=.01,
                        ax=ax,
                        norm=colors.SymLogNorm(linthresh=0.9),#colors.PowerNorm(gamma=1./5.)
                        cmap='Greens_r',#Blues_r
                        cbar_kws={"ticks":[0,1,10,1e2,1e3,1e4,1e5,1e6]},
                        xticklabels=3
                   )
            # Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
            plt.xticks(rotation=0)

            ax.set(ylabel='Type (ordered by amount)', xlabel='Creation Date')
            plt.savefig("heatmap--tipo-vs-ano.png",bbox_inches='tight')
run_plot('atividade_caracteristicas')
# df = df_original
df = df_original[df_original['bairro'] == 'BATEL']
def run(used_attribute='atividade_principal_agregada'):
    """
    function used only to clear some variables after execution
    """
    atividades_by_popularity = df \
        .groupby(used_attribute).size().reset_index(name='Size') \
        .sort_values('Size', ascending=False)[used_attribute].values
    count = pd.DataFrame()

    for atividade in atividades_by_popularity:
    # for atividade in ['CABELEIREIRO']:
        df_tmp = df[df[used_attribute] == atividade][df['data_inicio_atividade'] > '1980-01-01']
        df_tmp = df_tmp[['data_inicio_atividade']]
        df_tmp = df_tmp.append({'data_inicio_atividade':  pd.to_datetime('1970-01-01')}, ignore_index=True)
        df_tmp = df_tmp.append({'data_inicio_atividade':  pd.to_datetime('2017-01-01')}, ignore_index=True)
        df_tmp['Agg_Key'] = df_tmp['data_inicio_atividade'].values.astype('datetime64[Y]') # year
        # df_tmp['Agg_Key'] = df_tmp['data_inicio_atividade'].values.astype('datetime64[M]') # month
        df_tmp = df_tmp.groupby('Agg_Key').size().reset_index(name='Size') \
            .set_index('Agg_Key').resample('MS').asfreq().fillna(0)
        df_tmp[used_attribute] = atividade
        df_tmp = df_tmp.reset_index().sort_values(by='Agg_Key')
        values = df_tmp['Size'].values
        count[atividade] = values
    #     count_by_month_by_atividade = pd.concat([
    #         count_by_month_by_atividade,
    #         df_tmp.reset_index()
    #     ])
    # count_by_month_by_atividade = count_by_month_by_atividade.set_index([used_attribute, 'mes_inicio_atividade'])
    return count
count_by_month_by_atividade = run()
#count_by_month_by_atividade = run('atividade_caracteristicas')
'fim'
%matplotlib inline
f, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(count_by_month_by_atividade.corr(), ax=ax)
f.savefig("correlacao--tempo.png")
df = df_original[(df_original['bairro'] != 'INDICAÇÕES CANCELADAS') & (df_original['atividade_principal_agregada'] != 'X')]
# df = df_original[df_original['bairro'] == 'BATEL']
def run(used_attribute='atividade_principal_agregada', key_by='atividade'):
    df_tmp = df[df['data_inicio_atividade'] > '1980-01-01']

    df_tmp = (
        df_tmp.groupby([used_attribute,'bairro']).size().reset_index(name='counts') \
            .pivot_table(index='bairro', 
                         columns=[used_attribute], 
                         values='counts',
                         fill_value=0, 
                         aggfunc=np.sum
                         #aggfunc='count'
                          ).unstack() \
            .to_frame()
            .reset_index()
            .rename(columns={0:'counts'})
    #         [[used_attribute, 'counts']]
    #         .set_index(used_attribute)
    )
    
    count = pd.DataFrame()
    
    if key_by == 'atividade':
        df_tmp = df_tmp[[used_attribute, 'counts']]
        
        atividades_by_popularity = df \
            .groupby(used_attribute).size().reset_index(name='Size') \
            .sort_values('Size', ascending=False)[used_attribute].values

        for atividade in atividades_by_popularity:
            count[atividade] = df_tmp[df_tmp[used_attribute] == atividade]['counts'].values
        return count
    
    if key_by == 'bairro':
        df_tmp = df_tmp[['bairro', 'counts']]
        
        bairros_by_popularity = df \
            .groupby('bairro').size().reset_index(name='Size') \
            .sort_values('Size', ascending=False)['bairro'].values

        for bairro in bairros_by_popularity:
            count[bairro] = df_tmp[df_tmp['bairro'] == bairro]['counts'].values
        return count

# count_by_bairro_by_atividade = run()
count_by_bairro_by_atividade = run(key_by='bairro')

'fim'
df.groupby('bairro').size().reset_index(name='Size').sort_values('Size', ascending=False)['bairro'].values
                    # def run(used_attribute='atividade_principal_agregada'):
df_tmp = df[df['data_inicio_atividade'] > '1980-01-01']
df_tmp = df_tmp.append({'data_inicio_atividade':  pd.to_datetime('1970-01-01')}, ignore_index=True)
df_tmp = df_tmp.append({'data_inicio_atividade':  pd.to_datetime('2017-01-01')}, ignore_index=True)
df_tmp['Agg_Key'] = df_tmp['data_inicio_atividade'].values.astype('datetime64[Y]') # year

df_tmp = (
    df_tmp.groupby([used_attribute,'bairro','data_inicio_atividade']).size().reset_index(name='counts') \
        .pivot_table(index='bairro', 
                     columns=[used_attribute, 'data_inicio_atividade'], 
                     values='counts',
                     fill_value=0, 
                     aggfunc=np.sum
                     #aggfunc='count'
                      ).unstack() \
        .to_frame()
        .reset_index()
        .rename(columns={0:'counts'})
        [[used_attribute, 'counts']]
#         .set_index(used_attribute)
)

atividades_by_popularity = df \
    .groupby(used_attribute).size().reset_index(name='Size') \
    .sort_values('Size', ascending=False)[used_attribute].values
count = pd.DataFrame()

count = pd.DataFrame()
for atividade in atividades_by_popularity:
    count[atividade] = df_tmp[df_tmp[used_attribute] == atividade]['counts'].values
                    #     return count

                    # count_by_bairro_by_atividade = run()



'fim'
df_tmp
%matplotlib inline
f, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(count_by_bairro_by_atividade.corr(), ax=ax)
f.savefig("correlacao--bairro.png")
# clear variables
count_by_month_by_atividade = None
df = None
# %load_ext Cython
@numba.jit
def run():
    return 0

run()
df['day_gap_start'] = (df['data_inicio_atividade'].values.astype('datetime64[D]') - np.timedelta64(15,'D'))
df['day_gap_end'] = (df['data_inicio_atividade'].values.astype('datetime64[D]') + np.timedelta64(15,'D'))

def cond_merge(g, df_tmp):
    g = g[(g['data_inicio_atividade'] >= g['day_gap_start']) & (g['data_inicio_atividade'] <= g['day_gap_end'])]
    return g.groupby('day_gap_end').mean()
df2 = df.copy()
df2['the_day'] = df2['data_inicio_atividade']

df_tmp = df.merge(df2, on='bairro', how='outer')
df_tmp[(df_tmp['data_inicio_atividade'] != df_tmp['the_day']) &(df_tmp['the_day'] >= df_tmp['day_gap_start']) & (df_tmp['the_day'] <= df_tmp['day_gap_end'])] \
    [['data_inicio_atividade', 'the_day']]











































