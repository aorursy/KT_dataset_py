#hide

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sentences = [

    # Sita is a mischievous girl

    'சீதா ஒரு குறும்பு பெண்',

    # I remember my childhood

    'எனக்கு என் குழந்தைப் பருவம் நினைவிருக்கிறது',

    # India has successfully tested the Agni-5 missile for the fourth time from Abdul Kalam Island (Wheeler Island) in Odisha.

    'இந்தியா அக்னி-5 வகை ஏவுகணையை நான்காவது முறையாக வெற்றிகரமாக ஒடிசாவிலுள்ள அப்துல் கலாம் தீவிலிருந்து (வீலர் தீவு) சோதித்தது.',

    # The European Union's Galileo satellite system is in operation. It is believed to be the world's most accurate high-precision positioning system.

    'ஐரோப்பிய ஒன்றியத்தின் கலிலியோ செயற்கைகோள் அமைப்பு செயல்பாட்டுக்கு வந்துள்ளது. இது உலகின் மிக துல்லியமான செய்மதி இடஞ்சுட்டலாக இருக்கும் என நம்பப்படுகிறது.',

    # The factory has come in to operation after over 17 years of series of disruptions, including lack of cash.

    'இந்த தொழிற்சாலை பணபற்றாக்குறை முதலான பல்வேறு இடைஞ்சல்களை தாண்டி 17 ஆண்டுகள் கழித்தே செயல்பாட்டுக்கு வந்துள்ளது.',

    # Citizens, witnesses and warriors mourn the death of their king. It is up to the department to regret any loss.

    'தம் மன்னன் இறந்ததற்கு குடிமக்களும் சான்றோரும் வீரர்களும் வருந்திப் பாடுவது கையறுநிலை என்னும் துறையாகும். எந்த இழப்பையும் எண்ணி வருந்துவது கையறுநிலைத் துறைக்குரியது.',

    # The Poems from Sangam Tamil Literature portrays the trading feats of early Tamilian,Tamilians traded across seas and other countries

    'சங்கத்தமிழ்க் கவிதைகள் பழந்தமிழர்தம் வணிகச்சிறப்பைப் பறைசாற்றி நிற்கின்றன. தமிழர் கடல்கடந்து, அயல்நாடுகளிலும் வணிகம் செய்தனர் என்ற செய்திகளைச் சங்கப்பாடல்கள்வழி அறிகின்றோம்.',

    # Everyone stood up to call for a national flag at a school event.

    'பள்ளி நிகழ்ச்சி ஒன்றில் தேசியக் கொடி ஏற்றுமாறு அழைக்க அவரும் எழுந்தார் அனைவரும் எழுந்து நின்றனர்',

]
import sentencepiece as spm

from pathlib import Path

from IPython.core.display import display, HTML

from string import Template



sp = spm.SentencePieceProcessor()



TOK_PATH = '/kaggle/input/building-a-tokenizer-for-tamil-with-sentencepiece/tokenizer'



MODEL_PATHS = [p for p in Path(TOK_PATH).glob('*.model')]
#hide

# Just checking

MODEL_PATHS[0]
#hide

td_text = Template('<td>$text</td>')

tr_text = Template('<tr><td>$name</td> $li_elem</tr>')



# Utility to generate HTML text for a neat display

def tokenize_and_display_results(text):

    model_texts = []

    for model in MODEL_PATHS:

        # Load model

        sp.Load(str(model))

        

        # tokenize

        tok = sp.EncodeAsPieces(text)

        

        # Prepare html with string templates

        word_html = ''.join([td_text.substitute(text=word) for word in tok])

        list_html = tr_text.substitute(name=model.stem, li_elem=word_html)

        model_texts.append(list_html)

    

    return display(HTML('<table>' + ''.join(model_texts) + '</table>'))
tokenize_and_display_results(sentences[0])
tokenize_and_display_results(sentences[1])
tokenize_and_display_results(sentences[2])
tokenize_and_display_results(sentences[3])