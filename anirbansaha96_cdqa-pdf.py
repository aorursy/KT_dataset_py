!pip install cdqa
import os

import pandas as pd

from ast import literal_eval



from cdqa.utils.converters import pdf_converter

from cdqa.utils.filters import filter_paragraphs

from cdqa.pipeline import QAPipeline

from cdqa.utils.download import download_model

# Download model

download_model(model='bert-squad_1.1', dir='./models')
def download_pdf():

    import os

    import wget

    directory = './data/pdf/'

    models_url = [

      'https://docsmsftpdfs.blob.core.windows.net/guides/azure/azure-ops-guide.pdf'

    ]



    print('\nDownloading PDF files...')



    if not os.path.exists(directory):

        os.makedirs(directory)

    for url in models_url:

        wget.download(url=url, out=directory)



download_pdf()
df = pdf_converter(directory_path='./data/pdf/')
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)



# Fit Retriever to documents

cdqa_pipeline.fit_retriever(df=df)
query = 'What is the azure portal?'

prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))

print('answer: {}'.format(prediction[0]))

print('title: {}'.format(prediction[1]))

print('paragraph: {}'.format(prediction[2]))
query = 'What is An Azure Resource Manager template ?'

prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))

print('answer: {}'.format(prediction[0]))

print('title: {}'.format(prediction[1]))

print('paragraph: {}'.format(prediction[2]))
query = 'How can we automate?'

prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))

print('answer: {}'.format(prediction[0]))

print('title: {}'.format(prediction[1]))

print('paragraph: {}'.format(prediction[2]))
query = 'What is an Azure Resource Group?'

prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))

print('answer: {}'.format(prediction[0]))

print('title: {}'.format(prediction[1]))

print('paragraph: {}'.format(prediction[2]))
query = 'How do we use a pay-as-you-go model?'

prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))

print('answer: {}'.format(prediction[0]))

print('title: {}'.format(prediction[1]))

print('paragraph: {}'.format(prediction[2]))