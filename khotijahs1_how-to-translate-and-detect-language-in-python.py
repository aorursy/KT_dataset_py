#First, let's install it using pip:nstall 

!pip3 install googletrans
from googletrans import Translator, constants

from pprint import pprint
# init the Google API translator

translator = Translator()
# translate a Indonesia text to english text (by default)

translation = translator.translate("Selamat siang")

print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate Indonesian text to arabic for instance

translation = translator.translate("Selamat siang", dest="ar")

print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# specify source language

translation = translator.translate("Bagaimana kabarmu hari ini ?", src="id")

print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# print all translations and other data

pprint(translation.extra_data)
# translate more than a phrase

## (Indonesian to English)



sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="en")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate more than a phrase

## (Indonesian to Japanese)



sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="ja")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate more than a phrase 

#(Indonesian to Azerbaijan)

sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="az")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate more than a phrase 

#(Indonesian to Malay)



sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="ms")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate more than a phrase 

#(Indonesian to Turkish)



sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="tr")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# translate more than a phrase 

#(Indonesian to Hindi)



sentences = [

    "Halo Semua",

    "Apa kabar ?",

    "Apakah kamu bisa berbahsa Indonesia ?",

    "oke bagus!",

    "Senang berkenalan denganmu"

]

translations = translator.translate(sentences, dest="hi")

for translation in translations:

    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# detect a language

detection = translator.detect("नमस्ते दुनिया")

print("Language code:", detection.lang)

print("Confidence:", detection.confidence)
# detect a language

detection = translator.detect("kamu lagi apa ?")

print("Language code:", detection.lang)

print("Confidence:", detection.confidence)
# detect a language

detection = translator.detect("jam berapa kamu pergi ke sekolah  ?")

print("Language code:", detection.lang)

print("Confidence:", detection.confidence)
# detect a language

detection = translator.detect("sudah makan ?")

print("Language code:", detection.lang)

print("Confidence:", detection.confidence)
# detect a language

detection = translator.detect("Hepinize merhaba ?")

print("Language code:", detection.lang)

print("Confidence:", detection.confidence)
# print all available languages

print("Total supported languages:", len(constants.LANGUAGES))

print("Languages:")

pprint(constants.LANGUAGES)