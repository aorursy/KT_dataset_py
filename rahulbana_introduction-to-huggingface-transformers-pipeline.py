!pip install transformers
from transformers import pipeline
import numpy as np
pipeline("<task-name>")
pipeline("<task-name>", model="<model_name>")
pipeline('<task-name>', model='<model name>', tokenizer='<tokenizer_name>')
nlp_feature_extraction = pipeline('feature-extraction')
features = nlp_feature_extraction('The Book is on the table')

print("shape: ",np.array(features).shape)
print("\n")
print(features)
nlp_sentiment_analysis = pipeline('sentiment-analysis')
result = nlp_sentiment_analysis(task='I am happy after winning match.')

print(result)
nlp_ner = pipeline('ner')
result = nlp_ner(task='Hugging Face is a French company based in New-York.')

print(result)
nlp_qa = pipeline('question-answering')

desc = '''
Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.[28]

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.
'''
q = 'who developed python?'
result = nlp_qa(context=desc, question=q)

print(result)
nlp_fill_mask = pipeline('fill-mask')
result = nlp_fill_mask('The ball is on the <mask>')

print(result)
nlp_summary = pipeline('summarization')
txt = '''
Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.[28]

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.
'''
result = nlp_summary(txt, min_length=20, max_length=50)

print(result)
nlp_trans = pipeline('translation_en_to_fr')
txt = '''Happy New Year'''
result = nlp_trans(txt)

print(result)
nlp_text_generate = pipeline('text-generation')
txt = '''Today is a hot day. I am thinking '''
result = nlp_text_generate(txt, max_length=30)

print(result)
