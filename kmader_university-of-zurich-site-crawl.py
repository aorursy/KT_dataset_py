if False:
    !pip install -qq selenium
    !wget -q https://chromedriver.storage.googleapis.com/2.45/chromedriver_linux64.zip -O ../working/chromedriver.zip
    !cd ../working; unzip -q chromedriver.zip
    chrome_driver = '../working/chromedriver'
    from selenium import webdriver
    driver = webdriver.Chrome(chrome_driver)
else:
    import requests
    class RequestsDriver:
        def __init__(self):
            self.page_source=''
            self.error=False
        def get(self, url):
            try:
                self.page_source = requests.get(url).text
                self.error = False
            except Exception as e:
                self.page_source = 'Page cannot be loaded'
                self.error = True
        def quit(self):
            pass
    driver = RequestsDriver()
from bs4 import BeautifulSoup
from IPython.display import Image
import matplotlib.pyplot as plt
from six.moves.urllib_parse import urlparse, urljoin
import pandas as pd
import numpy as np
from bs4.element import Comment
from langdetect import detect
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True
parent_url = "https://www.uzh.ch/"
domain = urlparse(parent_url).netloc
parent_domain = 'uzh.ch'
def crawl_url(driver, url, black_list = None):
    """get contents and url from a page"""
    if black_list is None:
        black_list = []
    # Once the url is parsed, add it to crawled url list
    driver.get(url)
    if driver.error:
        text_output = driver.page_source
        title_txt = '404'
        urls = []
    else:
        html = driver.page_source.encode("utf-8")
        soup = BeautifulSoup(html, 'lxml')
        # parse text
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        text_output = u" ".join(t.strip() for t in visible_texts)
        title_txt = soup.title.string if soup.title is not None else ''
        
        urls = soup.findAll("a")
    
    url_list = []
    # Even if the url is not part of the same domain, it is still collected
    # But those urls not in the same domain are not parsed
    for a in urls:
        c_url = a.get("href")
        if (c_url):
            frag_parse = urlparse(c_url)
            if (len(frag_parse.path)>0) or (len(frag_parse.query)>0) or (len(frag_parse.netloc)>0):
                if len(frag_parse.netloc)==0:
                    c_url = urljoin(url, c_url)
                if (c_url not in url_list) and (c_url not in black_list):
                    url_list.append(c_url)
    
    return set(url_list), (title_txt, text_output)
links, (page_title, page_text) = crawl_url(driver, 'https://www.fartyfacybagoon.ixm')
print('Links:', len(links))
print('Language:', detect(page_text), 'Title:', page_title, 'Content:', page_text[:80])
links, (page_title, page_text) = crawl_url(driver, parent_url)
print('Links:', len(links))
print('Language:', detect(page_text), 'Title:', page_title, 'Content:', page_text[:80])
link_queue = {parent_url}
crawled_urls = dict()
edge_list = set()
search_depth = 0
MAX_ARTICLES = 5000
while len(link_queue)>0:
    if len(crawled_urls)>MAX_ARTICLES:
        break
    finished_pages = len(crawled_urls)
    next_links = set()
    # breadth first crawl
    for c_link in link_queue:
        if c_link not in crawled_urls:
            links, page_text = crawl_url(driver, c_link)
            crawled_urls[c_link] = page_text
            if len(crawled_urls)>MAX_ARTICLES:
                break
            for r_link in links:
                edge_list.add((c_link, r_link))
                if parent_domain in urlparse(r_link).netloc:
                    next_links.add(r_link)
    print(finished_pages, ':', len(link_queue), '->',  len(next_links))
    link_queue = next_links.copy()
    search_depth += 1
# Finally quit the browser
driver.quit()
print("Edges", len(edge_list))
print("URLs Crawled", len(crawled_urls))
webtext_df = pd.DataFrame([{'url': a, 'contents': contents, 
               'title': title} 
              for a,(title, contents) in crawled_urls.items()])
def _soft_detect(in_text):
    try:
        return detect(in_text)
    except:
        return ''
webtext_df['language'] = webtext_df['contents'].map(_soft_detect)
webtext_df['clean_url'] = webtext_df['url'].map(lambda x: x.replace('www.', ''))
webtext_df['domain'] = webtext_df['clean_url'].map(lambda x: urlparse(x).netloc)
webtext_df['path'] = webtext_df['clean_url'].map(lambda x: urlparse(x).path)
webtext_df.to_json('full_crawl.json')
webtext_df.sample(3)
webtext_df.groupby('language').size().plot.bar()
edge_df = pd.DataFrame([{'src': a, 'dst': b} for a, b in edge_list])
edge_df['src'] = edge_df['src'].map(lambda x: x.replace('www.', ''))
edge_df['dst'] = edge_df['dst'].map(lambda x: x.replace('www.', ''))
edge_df['src_parse'] = edge_df['src'].map(urlparse)
edge_df['dst_parse'] = edge_df['dst'].map(urlparse)
for base in ['src', 'dst']:
    for node in ['netloc', 'path', 'params', 'query']:
        edge_df[f'{base}_{node}'] = edge_df[f'{base}_parse'].map(lambda x: getattr(x, node))
edge_df.drop(['src_parse', 'dst_parse'], axis=1).to_csv('edges.csv')
edge_df.sample(3)
domain_df = pd.DataFrame({'netloc': edge_df['src_netloc'].values.tolist()+
                          edge_df['dst_netloc'].values.tolist()}).\
    groupby('netloc').size().reset_index(name='count').\
    sort_values('count', ascending=False)
domain_df = domain_df[domain_df.iloc[:, 0].map(len)>0]
top_domains_df = domain_df.head(25)
top_domains_df
dom_edge_df = edge_df.\
    groupby(['src_netloc', 'dst_netloc']).\
    size().\
    reset_index(name='count').\
    sort_values('count', ascending=False)
graph_df = pd.merge(
    pd.merge(dom_edge_df, 
             top_domains_df, 
             left_on='src_netloc', 
             right_on='netloc', 
             suffixes=('', '_src')).drop(['netloc'],axis=1),
    top_domains_df, 
    left_on='dst_netloc', 
    right_on='netloc', 
    suffixes=('', '_dst')
).drop(['netloc'],axis=1)
graph_df.sample(3)
import pydot
import networkx as nx
g = nx.MultiDiGraph()
for _, c_row in graph_df.iterrows():
    a_name = c_row.iloc[0]
    b_name = c_row.iloc[1]
    g.add_edge(a_name, b_name, weight=c_row.iloc[2])
fig, ax1 = plt.subplots(1, 1, figsize = (20, 15))
ax1.axis('off')
#pos = nx.nx_agraph.graphviz_layout(g)
pos = nx.spring_layout(g, iterations=20)
pos = nx.kamada_kawai_layout(g)
#pos = nx.circular_layout(g)
#pos = nx.spectral_layout(g)
nx.draw_networkx_edges(g, 
                       pos, 
                       alpha=0.3, 
                       ax=ax1, 
                       edge_color='g',
                      width = np.log2(graph_df.iloc[:, 2].values))
nx.draw_networkx_nodes(g, 
                       pos, 
                       node_color='r', 
                       alpha=0.4, 
                       ax=ax1)
#nx.draw_networkx_edges(g, pos, alpha=0.4, node_size=0, width=1, edge_color='k', ax=ax1)
nx.draw_networkx_labels(g, pos, fontsize=14, ax=ax1);
d = nx.drawing.nx_pydot.to_pydot(g)
Image(d.create_png())