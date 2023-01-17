%%time

!pip uninstall -q neuralcoref -y > /dev/null
!pip install -q neuralcoref --no-binary neuralcoref > /dev/null

!pip uninstall -q spacy -y > /dev/null
!pip install -q -U spacy==2.1.0 > /dev/null
!python -m spacy download en > /dev/null
import json, random
from tqdm.notebook import tqdm
from urllib.parse import quote

# Load your usual SpaCy model (one of SpaCy English models)
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)
text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

doc = nlp(text)
displacy.render(doc, style="ent")
doc = nlp('Angela lives in Boston. She is quite happy in that city.')
for ent in doc.ents:
    print(ent._.coref_cluster)
with open('../input/this-american-life-podcast-transcriptsalignments/test-transcripts-aligned.json', 'r') as f:
    transcripts = json.load(f)
episode_list = []

for episode in transcripts:
    episode_list.append(episode)

print(episode_list)
sample_episode = 'ep-120'
for segment in transcripts[sample_episode][:6]:
    print(segment['speaker'], ": ", segment['utterance'], "\n")
sample_episode = 'ep-648'
for segment in transcripts[sample_episode][:6]:
    print(segment['speaker'], ": ", segment['utterance'], "\n")
text = transcripts['ep-648'][0]["utterance"]
doc = nlp(text)

displacy.render(doc, style="ent")
text = transcripts[random.choice(episode_list)][0]["utterance"]
doc = nlp(text)

displacy.render(doc, style="ent")
episode = random.choice(episode_list)
print("Episode: ", episode)
text = transcripts[episode][0]["utterance"]
url_coref = "https://huggingface.co/coref/?text=" + quote(text)
print("Text: ", text, "\n")
doc = nlp(text)


for ent in doc.ents:
    if ent._.coref_cluster:
        print(ent._.coref_cluster)
        
print("\n Navigate to the below URL for visualizing Coreference Resolution:\n", url_coref)