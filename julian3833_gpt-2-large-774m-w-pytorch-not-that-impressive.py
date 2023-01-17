!pip install transformers
import torch

import numpy as np

    

def fix_randomness():

    seed = 123

    np.random.seed(seed)

    torch.random.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    

fix_randomness()
import logging

from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.getLogger().setLevel(logging.CRITICAL) # Disable an annoying warning for now



SAMPLE_INPUTS = [

    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",

    "A train carriage containing controlled nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown.",

    "Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.",

    "We’ve trained a large language model called GPT-2 that generates realistic paragraphs of text, while also exhibiting zero shot generalization on tasks like machine translation, question answering, reading comprehension, and summarization - problems usually approached by using training datasets and models designed explicitly for these tasks.",

    "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",

    "For today’s homework assignment, please describe the reasons for the US Civil War.",

    "John F. Kennedy was just elected President of the United States after rising from the grave decades after his assassination. Due to miraculous developments in nanotechnology, Kennedy’s brain was rebuilt from his remains and installed in the control center of a state-of-the art humanoid robot. Below is a transcript of his acceptance speech.",

    "Recycling is good for the world.\n\nNO! YOU COULD NOT BE MORE WRONG!!"

    ]



def get_tokenizer_and_model(model_id):

    assert model_id in ['gpt2', 'gpt2-medium', 'gpt2-large']

    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    model = GPT2LMHeadModel.from_pretrained(model_id).eval().to('cuda')

    return tokenizer, model



def stoi(tokenizer, text):

    indexed_tokens = tokenizer.encode(text)

    tokens = torch.tensor([indexed_tokens]).to('cuda')

    return tokens



def generate_next_token(model, tokens):

    with torch.no_grad():

        outputs = model(tokens)

        predictions = outputs[0]

    predicted_token = torch.argmax(predictions[0, -1, :]).item()

    return predicted_token



def add_token(tokens, token):

    token_in_dimensions_and_gpu = torch.tensor([[token]]).to('cuda')

    return torch.cat([tokens, token_in_dimensions_and_gpu], dim=1)



def add_n_tokens(model, tokens, n_tokens):

    generated = []

    for _ in range(n_tokens):

        new = generate_next_token(model, tokens)

        tokens = add_token(tokens, new)

        generated.append(new)

    return tokens, generated



def remove_repetitions(text):

    first_ocurrences = []

    for sentence in text.split("."):

        if sentence not in first_ocurrences:

            first_ocurrences.append(sentence)

    return '.'.join(first_ocurrences)



def trim_last_sentence(text):

    return text[:text.rfind(".")+1]



def postprocess(text):

    return trim_last_sentence(remove_repetitions(text))



def generate(text, n_words=10, model_id='gpt2'):

    print(f"MODEL: {model_id}")

    

    tokenizer, model = get_tokenizer_and_model(model_id)

    tokens = stoi(tokenizer, text)

    tokens, generated = add_n_tokens(model, tokens, n_words)

    

    generated_text = postprocess(tokenizer.decode(generated))

    

    print(f"INPUT: {text}")

    print(f"OUTPUT: {generated_text}\n")
%%time

text = "Hello GPT-2, how are you doing?"

n_words = 20



generate(text, n_words)

generate(text, n_words, 'gpt2-medium')

#generate(text, n_words, 'gpt2-large')
def benchmark(model_id, n_words, texts=SAMPLE_INPUTS):

    

    print(f"{model_id} with n_words={n_words}\n=========================")

    tokenizer, model = get_tokenizer_and_model(model_id)

    

    for text in texts:

        tokens = stoi(tokenizer, text)

        tokens, generated = add_n_tokens(model, tokens, n_words)

        generated_text = postprocess(tokenizer.decode(generated))

    

        print("INPUT: {}".format(text.replace('\n', '')))

        print("OUTPUT: {}".format(generated_text.replace('\n', '')))

        print("\n====\n")

%%time

benchmark('gpt2', n_words=200)
%%time

benchmark('gpt2-medium', n_words=200)
#%%time

#benchmark('gpt2-large', n_words=200)
%%time 



ww2 = """World War II (often abbreviated to WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945. The vast majority of the world's countries—including all the great powers—eventually formed two opposing military alliances: the Allies and the Axis. A state of total war emerged, directly involving more than 100 million people from over 30 countries. The major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities, most of whom were civilians in the Soviet Union and China. It included massacres, the genocide of the Holocaust, strategic bombing, premeditated death from starvation and disease, and the only use of nuclear weapons in war.[1][2][3][4]"""

#benchmark('gpt2-large', n_words=200, texts=[ww2])
benchmark('gpt2-medium', n_words=200, texts=[ww2])