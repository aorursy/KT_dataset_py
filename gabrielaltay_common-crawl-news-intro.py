!pip install warcio
import os

from bs4 import BeautifulSoup                                                                            

from warcio.archiveiterator import ArchiveIterator 
file_path = os.path.join(

    '/kaggle/input',

    'common-crawl-news-2020011021203700310',

    'CC-NEWS-20200110212037-00310.warc')
uris_and_titles = []

with open(file_path, 'rb') as stream:

    for record in ArchiveIterator(stream):

        

        if record.rec_type == 'request':

            pass

        

        if record.rec_type == 'response':

            warc_target_uri = record.rec_headers.get_header('WARC-Target-URI')

            article_html = record.content_stream().read()

            soup = BeautifulSoup(article_html, 'html.parser')

            title = soup.title.string

            uris_and_titles.append((warc_target_uri, title))

            text = soup.get_text()

            lines = text.split('\n')

            

        if len(uris_and_titles) > 10:

            break
uris_and_titles
lines