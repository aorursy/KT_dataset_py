!pip install transformers
from __future__ import print_function

import ipywidgets as widgets

from transformers import pipeline
nlp_sentence_classif = pipeline('sentiment-analysis')

nlp_sentence_classif('Such a nice weather outside !')
nlp_token_class = pipeline('ner')

nlp_token_class('Hugging Face is a French company based in New-York.')
nlp_qa = pipeline('question-answering')

nlp_qa(context='Hugging Face is a French company based in New-York.', question='Where is based Hugging Face ?')
nlp_fill = pipeline('fill-mask')

nlp_fill('Hugging Face is a French company based in <mask>')
import numpy as np

nlp_features = pipeline('feature-extraction')

output = nlp_features('Hugging Face is a French company based in Paris')

np.array(output).shape   # (Samples, Tokens, Vector Size)

task = widgets.Dropdown(

    options=['sentiment-analysis', 'ner', 'fill_mask'],

    value='ner',

    description='Task:',

    disabled=False

)



input = widgets.Text(

    value='',

    placeholder='Enter something',

    description='Your input:',

    disabled=False

)



def forward(_):

    if len(input.value) > 0: 

        if task.value == 'ner':

            output = nlp_token_class(input.value)

        elif task.value == 'sentiment-analysis':

            output = nlp_sentence_classif(input.value)

        else:

            if input.value.find('<mask>') == -1:

                output = nlp_fill(input.value + ' <mask>')

            else:

                output = nlp_fill(input.value)                

        print(output)



input.on_submit(forward)

display(task, input)
context = widgets.Textarea(

    value='Einstein is famous for the general theory of relativity',

    placeholder='Enter something',

    description='Context:',

    disabled=False

)



query = widgets.Text(

    value='Why is Einstein famous for ?',

    placeholder='Enter something',

    description='Question:',

    disabled=False

)



def forward(_):

    if len(context.value) > 0 and len(query.value) > 0: 

        output = nlp_qa(question=query.value, context=context.value)            

        print(output)



query.on_submit(forward)

display(context, query)