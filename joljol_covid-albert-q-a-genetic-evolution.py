import torch 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU recommended

# Loading custom pre-trained ALBERT model already fine-tuned to SQuAD 2.0
import transformers
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForQuestionAnswering.from_pretrained(
    '/kaggle/input' \
    '/nlp-albert-models-fine-tuned-for-squad-20'\
    '/albert-base-v2-tuned-for-squad-2.0').to(device)

# Loading the CORD-19 dataset and pre-processing
import pandas as pd
data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv',
                   keep_default_na=False)
data = data[data['abstract']!=''] \
       .reset_index(drop=True) # Remove rows with no abstracts
transformers.__version__
import numpy as np
from tensorflow.keras.utils import Progbar

def inference_ALBERT(question):
    
    spans, scores, token_ids = [], [], []
    
    # Iterating over all CORD-19 articles and perform model inference
    progress_bar = Progbar(len(data))
    for i in range(len(data)):
        if i % 500 == 0:
            progress_bar.update(i)
        text = data['abstract'][i]
        input_ids = tokenizer.encode(question, text)
        
        # We have token limit of 512, so truncate if needed
        if len(input_ids) > 512:
            input_ids, token_type_ids = \
                input_ids[:511] + [3], token_type_ids[:512]
                # [3] is the SEP token
        
        token_type_ids = [0 if i <= input_ids.index(3) 
                          else 1 for i in range(len(input_ids))]

        # Preparing the tensors for feeding into model
        input_ids_tensor = torch.tensor([input_ids]).to(device)
        token_type_ids_tensor = torch.tensor([token_type_ids]).to(device)
        
        # Performing model inference
        start_scores, end_scores = \
            model(input_ids_tensor, 
                  token_type_ids=token_type_ids_tensor)
        
        # Releasing GPU memory by moving each tensor back to CPU
        # If GPU is not used, this step is uncessary but won't give error
        input_ids_tensor, token_type_ids_tensor, start_scores, end_scores = \
            tuple(map(lambda x: x.to('cpu').detach().numpy(), 
                     (input_ids_tensor, token_type_ids_tensor, \
                      start_scores, end_scores)))
        # Let me know if there's an easier way to do this, as I mostly work
        # with tensorflow and I'm not very familiar with Keras

        # Appending results to the corresponding lists
        # Spans are the indices of the start and end of the answer
        spans.append( [start_scores.argmax(), end_scores.argmax()+1] )
        # Scores are the "confidence" level in the start and end
        scores.append( [start_scores.max(), end_scores.max()] )
        token_ids.append( input_ids )

    spans = np.array(spans, dtype='int')
    scores = np.array(scores)
    
    return spans, scores, token_ids
# Define a helper function to directly convert token IDs to string
convert_to_str = lambda token_ids: \
    tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(token_ids))

from IPython.display import display, HTML

def display_results(spans, scores, token_ids, first_n_entries=15,
                    max_disp_len=100):
    
    display(HTML(
        'Model output (<text style=color:red>red font</text> '\
        'highlights the answer predicted by ALBERT NLP model)'\
        ))
    
    # We first sort the results based on the confidence in either the 
    # start or end index of the answer, whichever is smaller
    min_scores = scores.min(axis=1) 
    sorted_idx = (-min_scores).argsort() # Descending order
    
    counter = 0    
    for idx in sorted_idx:
        
        # Stop if first_n_entries papers have been displayed
        if counter >= first_n_entries:
            break
        
        # If the span is empty, the model prdicts no answer exists 
        # from the article. In rare cases, the end is smaller than
        # the start. Both will be skipped
        if spans[idx,0] == 0 or spans[idx,1] == 0 or \
            spans[idx,1]<=spans[idx,0]:
            continue

        # Obtaining the start and end token indices of answer
        start, end = spans[idx, :]

        abstract = data['abstract'][idx]
        abstract_highlight = convert_to_str(token_ids[idx][start:end])
        
        # If we cannot fully convert tokens to original text,
        # we then use the detokenized text (lower cased)
        # Otherwise it would be best to have the original text,
        # because there's lots of formatting especially in bio articles
        start = abstract.lower().find(abstract_highlight)
        if start == -1:
            abstract = convert_to_str(token_ids[idx]
                                      [token_ids[idx].index(3)+1:])
            start = abstract.find(abstract_highlight)
            end = start + len(abstract_highlight)
            abstract = abstract[:-5] # to remove the [SEP] token in the end
        else:
            end = start + len(abstract_highlight)
            abstract_highlight = abstract[start:end]
        abstract_before_highlight, abstract_after_highlight = \
            abstract[: start], \
            abstract[end : ]
    
        # Putting information in HTML format
        html_str = f'<b>({counter+1}) {data["title"][idx]}</b><br>' + \
                   f'Confidence: {scores[idx].min():.2f} | ' + \
                   f'<i>{data["journal"][idx]}</i> | ' + \
                   f'{data["publish_time"][idx]} | ' + \
                   f'<a href={data["url"][idx]}>{data["doi"][idx]}</a>' + \
                   '<p style=line-height:1.1><font size=2>' + \
                   abstract_before_highlight + \
                   '<text style=color:red>%s</text>'%abstract_highlight + \
                   abstract_after_highlight + '</font></p>'
        
        display(HTML(html_str))
        
        counter += 1

# Combining the inference function and the display function into one
def inference_ALBERT_and_display_results(question, 
                                         first_n_entries=15,
                                         max_disp_len=100):
    
    spans, scores, token_ids = inference_ALBERT(question)
    display_results(spans, scores, token_ids, 
                    first_n_entries, max_disp_len)
inference_ALBERT_and_display_results(
    'Real-time tracking of whole genomes and a mechanism '
    'for coordinating the rapid dissemination of that information '
    'to inform the development of diagnostics and therapeutics '
    'and to track variations of the virus over time ')
inference_ALBERT_and_display_results(
    'Access to geographic and temporal diverse sample sets '
    'to understand geographic distribution and genomic '
    'differences, and determine whether there is more '
    'than one strain in circulation, leveraging '
    'multi-lateral agreements such as the Nagoya Protocol ')
inference_ALBERT_and_display_results(
    'Evidence of whether farmers are infected, and '
    'whether farmers could have played a role in the origin ')
inference_ALBERT_and_display_results(
    'Surveillance of mixed wildlife- livestock farms for '
    'SARS-CoV-2 and other coronaviruses in Southeast Asia ')
inference_ALBERT_and_display_results(
    'What is the host range for the coronavirus pathogen?')
inference_ALBERT_and_display_results(
    'Animal hosts and any evidence of continued '
    'spill-over to humansv')
inference_ALBERT_and_display_results(
    'What are the socioeconomic and behavioral risk for '
    'continuous spill over from animals to humans?')
inference_ALBERT_and_display_results(
    'What are sustainable risk reduction strategies?')
print('Done!')