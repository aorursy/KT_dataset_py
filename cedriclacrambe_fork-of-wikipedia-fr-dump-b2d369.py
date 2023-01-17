# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# !pip install -q  mediawiki-parser

!pip install git+https://github.com/peter17/pijnu.git

!pip install -U  mwxml  wikitextparser git+https://github.com/peter17/mediawiki-parser.git

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pypandoc

import wikitextparser as wtp

import os

print(os.listdir("../input"))

import json,requests,glob

import gensim

import zipfile

import lzma

import mwxml 

from mwxml import Dump, Page

import smart_open

import re

# Any results you write to the current directory are saved as output.


#! wget    https://dumps.wikimedia.freemirror.org/frwikisource/20190301/frwikisource-20190301-pages-articles-multistream.xml.bz2 -P /tmp/

#! wget    ftp://ftpmirror.your.org/pub/wikimedia/dumps/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2 -P /tmp/

   

# ! wget    https://dumps.wikimedia.freemirror.org/frwikisource/latest/frwikisource-latest-pages-articles-multistream.xml.bz2 -P /tmp/
import mediawiki_parser

import mediawiki_parser.preprocessor, mediawiki_parser.text

def wikiparse(text):

    templates = {}





    preprocessor = mediawiki_parser.preprocessor.make_parser(templates)



    parser =  mediawiki_parser.text.make_parser()



    preprocessed_text = preprocessor.parse(text)

    output = parser.parse(preprocessed_text)

    return output.leaves()
def wiki_article_generator(source,len_threshold=50):

        with smart_open.open(source) as f:



            dump = Dump.from_file(f)

            for page in dump.pages:

                # Iterate through a page's revisions

                for revision in page:

                    pass



                text=revision.text

                if text is not None and len(text)>len_threshold:



                    text=text.strip()

                    l=len(text)

                    if  any( t in page.title   for t in [".djvu",".jpg",".png"]):#re.fullmatch(r".*?\.\w{2,5}/\d*",page.title):

    #                     print("file",[page.title])

                        continue

                    title=page.title.replace("/"," - ")

                    title=title.replace("\\"," - ")

                    import pypandoc

                    try:

                        text=pypandoc.convert_text(text,'plain','mediawiki')

                        

                    except:

                        try:

                            text=wikiparse(text)

                        except:

                            pass

                    yield (page.title,text)







 
from mwxml import Dump, Page

import smart_open

import re



source="/tmp/frwiki-latest-pages-articles-multistream.xml.bz2"

source="https://dumps.wikimedia.freemirror.org/frwikisource/latest/frwikisource-latest-pages-articles-multistream.xml.bz2"











def extract_wikimedia_texts_zip(source,destination):

    with zipfile.ZipFile(destination, mode="a",compression=zipfile.ZIP_DEFLATED) as zf:

        n=1

        

        with smart_open.open(source) as f:



            dump = Dump.from_file(f)

            for page in dump.pages:

                # Iterate through a page's revisions

                for revision in page:

                    pass



                text=revision.text

                if text is not None and len(text)>50:



                    text=text.strip()

                    l=len(text)

                    if  any( t in page.title   for t in [".djvu",".jpg",".png"]):#re.fullmatch(r".*?\.\w{2,5}/\d*",page.title):

    #                     print("file",[page.title])

                        continue

                    title=page.title.replace("/"," - ")

                    title=title.replace("\\"," - ")



                    zf.writestr(f"{title}- articles{n}.txt",text)

                    n+=1

                st=os.statvfs(".")

                disk_free=st.f_bavail*st.f_bsize

                if n%5000==0:

                    print(n,disk_free,page.title)

                if disk_free<5e6:

                    print("disk full")

                    break        

!ls -l

!rm *.zip
sources=["https://dumps.wikimedia.freemirror.org/frwikisource/latest/frwikisource-latest-pages-articles-multistream.xml.bz2",

          "https://dumps.wikimedia.freemirror.org/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2",

         "https://dumps.wikimedia.freemirror.org/frwikiversity/latest/frwikiversity-latest-pages-articles.xml.bz2",

         "https://dumps.wikimedia.freemirror.org/frwikibooks/latest/frwikibooks-latest-pages-articles.xml.bz2",

         "https://dumps.wikimedia.freemirror.org/frwikivoyage/latest/frwikivoyage-latest-pages-articles.xml.bz2"

         

        

        ]
import urllib.parse

import os





for source in sources:

    n=1

    destination =os.path.basename(urllib.parse.urlparse(sources[0]).path).split("-")[0]+".zip"

#     extract_wikimedia_texts_zip(source,destination)

    with zipfile.ZipFile(destination, mode="a",compression=zipfile.ZIP_DEFLATED) as zf:

        print ("writing",destination)

        for title,text in wiki_article_generator(source):

            title=title.replace("/"," - ")

            title=title.replace("\\"," - ")

            output_name=f"{title}.txt" #- article {n}



            zf.writestr(output_name,text)

            n+=1



            st=os.statvfs(".")

            disk_free=st.f_bavail*st.f_bsize

            if n%5000==0:

                print(n,disk_free,title)

            if disk_free<4e6:

                print("disk full")

                break  



    

    if disk_free<400e6:

        print("disk full")

        break     
text
for source in sources:

    n=1

    for title,text in wiki_article_generator(source):

        try:

            print (title)

            if n%10==0:

                print (text)

            n+=1

            if n >200:

                break



        except:

            pass