#@title Installing Dependencies
!pip install --quiet tensorflow_text
#@title Imports
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.WARN)
#@title { run: "auto" }
language = "uk" #@param ["en", "ar", "zh-cn", "zh-tw", "nl", "fr", "de", "it", "ja", "ko", "pl", "pt", "ru", "es", "th", "tr", "bg", "ca", "cs", "da", "el", "et", "fa", "fi", "he", "hi", "hr", "hu", "id", "lt", "lv", "ms", "no", "ro", "sk", "sl", "sr", "sv", "tl", "uk", "vi", "multilingual-64k", "multilingual-128k"]
hub_module = "https://tfhub.dev/google/wiki40b-lm-{}/1".format(language)
max_gen_len = 20 #@param

print("Using the {} model to generate sequences of max length {}.".format(hub_module, max_gen_len))
#@title Load the language model pieces
g = tf.Graph()
n_layer = 12
model_dim = 768

with g.as_default():
  text = tf.placeholder(dtype=tf.string, shape=(1,))

  # Load the pretrained model from TF-Hub
  module = hub.Module(hub_module)

  # Get the word embeddings, activations at each layer, negative log likelihood
  # of the text, and calculate the perplexity.
  embeddings = module(dict(text=text), signature="word_embeddings", as_dict=True)["word_embeddings"]
  activations = module(dict(text=text), signature="activations", as_dict=True)["activations"]
  neg_log_likelihood = module(dict(text=text), signature="neg_log_likelihood", as_dict=True)["neg_log_likelihood"]
  ppl = tf.exp(tf.reduce_mean(neg_log_likelihood, axis=1))
#@title Construct the per-token generation graph
def feedforward_step(module, inputs, mems):
  """Generate one step."""
  # Set up the input dict for one step of generation
  inputs = tf.dtypes.cast(inputs, tf.int64)
  generation_input_dict = dict(input_tokens=inputs)
  mems_dict = {"mem_{}".format(i): mems[i] for i in range(n_layer)}
  generation_input_dict.update(mems_dict)

  # Generate the tokens from the language model
  generation_outputs = module(generation_input_dict, signature="prediction", as_dict=True)

  # Get the probablities and the inputs for the next steps
  probs = generation_outputs["probs"]
  new_mems = [generation_outputs["new_mem_{}".format(i)] for i in range(n_layer)]

  return probs, new_mems
#@title Build the statically unrolled graph for `max_gen_len` tokens
with g.as_default():
  # Tokenization with the sentencepiece model.
  token_ids = module(dict(text=text), signature="tokenization", as_dict=True)["token_ids"]
  inputs_np = token_ids
  # Generate text by statically unrolling the computational graph
  mems_np = [np.zeros([1, 0, model_dim], dtype=np.float32) for _ in range(n_layer)]

  # Generate up to `max_gen_len` tokens
  sampled_ids = []
  for step in range(max_gen_len):
    probs, mems_np = feedforward_step(module, inputs_np, mems_np)
    sampled_id = tf.random.categorical(tf.math.log(probs[0]), num_samples=1, dtype=tf.int32)
    sampled_id = tf.squeeze(sampled_id)
    sampled_ids.append(sampled_id)
    inputs_np = tf.reshape(sampled_id, [1, 1])

  # Transform the ids into text
  sampled_ids = tf.expand_dims(sampled_ids, axis=0)
  generated_text = module(dict(token_ids=sampled_ids), signature="detokenization", as_dict=True)["text"]

  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#@title Predefined Seeds
lang_to_seed = {"en": "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada.",
                "ar": "\n_START_ARTICLE_\n?????????????? ??????\n_START_SECTION_\n???????????? ?????????????? \n_START_PARAGRAPH_\n???????? ?????????????? ?????? ???? ?????????????? ???? ?????????????? ???????????? ?????????? ?????????? ???? ???? ???????? ?????????? ?????????? ?????????? ???????? ???????????? ????????????. ?????????? ???????? ?????????? ???????? ???????? ???????????? ???? ?????????????? ??????????????. ?????? ?????????????? ?????????? ???? ?????????????????? ?????????????????? ???????????? ?????? ?????????? ???? ???????? ?????????????? . ?????? ???? ?????????????? ?????? ???????????? ???????? ?????? ?????????? ?????? ???? ?????????????? ?????? ???????? ?????????? ???? ?????????? ???????????????? ???????????????? ???? ????????????????",
                "zh-cn": "\n_START_ARTICLE_\n????????????\n_START_SECTION_\n??????????????????????????????\n_START_PARAGRAPH_\n???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????17?????????????????????????????????????????????????????????????????????????????????",
                "zh-tw": "\n_START_ARTICLE_\n??????\n_START_PARAGRAPH_\n??????????????????Houssen????????????[us??n]????????????Hausen?????????????????????H??se??????????????????????????????????????????????????????????????????????????????-??????????????????Colmar-Ribeauvill??????????????????????????Colmar-2????????????????????????6.7???????????????2009??????????????????",
                "nl": "\n_START_ARTICLE_\n1001 vrouwen uit de Nederlandse geschiedenis\n_START_SECTION_\nSelectie van vrouwen\n_START_PARAGRAPH_\nDe 'oudste' biografie in het boek is gewijd aan de beschermheilige",
                "fr": "\n_START_ARTICLE_\n???\n_START_SECTION_\nUtilisation\n_START_PARAGRAPH_\nLe d insulaire est utilis?? comme lettre additionnelle dans l?????dition de 1941 du recueil de chroniques galloises Brut y Tywysogion",
                "de": "\n_START_ARTICLE_\n??nal Demirk??ran\n_START_SECTION_\nLaufbahn\n_START_PARAGRAPH_\nDemirk??ran deb??tierte als junges Talent am 25. September 1999 im Ausw??rtsspiel des SSV Ulm 1846 bei Werder Bremen (2:2) in der Bundesliga, als er kurz",
                "it": "\n_START_ARTICLE_\n28th Street (linea IRT Lexington Avenue)\n_START_SECTION_\nStoria\n_START_PARAGRAPH_\nLa stazione, i cui lavori di costruzione ebbero inizio nel 1900, venne aperta il 27 ottobre 1904, come",
                "ja": "\n_START_ARTICLE_\n?????????????????????show'05 ??????????????????\n_START_SECTION_\n??????\n_START_PARAGRAPH_\n?????????????????????SHOW??????????????????????????????????????????????????????????????????MC?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????",
                "ko": "\n_START_ARTICLE_\n??????, Op. 9 (??????)\n_START_SECTION_\n?????? 3??? ?????????\n_START_PARAGRAPH_\n????????? ?????? 3?????? ????????? ????????? (A-B-A)????????? ????????? ??????. ??? ????????? ???????????????(Allegretto)??? ???????????? ???????????? ????????? ??? ???????????? ???????????? ????????????",
                "pl": "\n_START_ARTICLE_\nAK-176\n_START_SECTION_\nHistoria\n_START_PARAGRAPH_\nPod koniec lat 60 XX w. w ZSRR dostrze??ono potrzeb?? posiadania lekkiej armaty uniwersalnej ??redniego kalibru o stosunkowo du??ej mocy ogniowej, kt??ra",
                "pt": "\n_START_ARTICLE_\n??cido ribonucleico\n_START_SECTION_\nIntermedi??rio da transfer??ncia de informa????o\n_START_PARAGRAPH_\nEm 1957 Elliot Volkin e Lawrence Astrachan fizeram uma observa????o significativa. Eles descobriram que uma das mais marcantes mudan??as",
                "ru": "\n_START_ARTICLE_\n??????????????, ????????\n_START_SECTION_\n?????????????? ??????????????\n_START_PARAGRAPH_\n?????????????? ?????????????? ?? ???????????????? ?????????????????? ?? 12 ??????. ?? 2014 ???????? ???????????????? ???? ???????????? ??????????????, ?????? ???????????? ???????????????????? ????????????. ?? ???????????? 2015/2016 ?????????????????? ?? ?????????????? ???????????????? ??????????????. 27 ???????????????? 2015 ???????? ??????????????????????",
                "es": "\n_START_ARTICLE_\n(200012) 2007 LK20\n_START_SECTION_\nDesignaci??n y nombre\n_START_PARAGRAPH_\nDesignado provisionalmente como 2007 LK20.\n_START_SECTION_\nCaracter??sticas orbitales\n_START_PARAGRAPH_\n2007 LK20",
                "th": "\n_START_ARTICLE_\n???????????????????????????????????????????????????????????????????????????????????????\n_START_SECTION_\n????????????????????? ????????????????????????\n_START_PARAGRAPH_\n?????????????????? 20 ????????????????????? 2561 ????????????????????? ???????????????????????? ???????????????????????????????????????????????????????????????????????????????????? ?????????????????????????????????????????????????????????????????????????????????????????? 9 (???????????????????????????????????????????????????????????????????????? 3) ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????",
                "tr": "\n_START_ARTICLE_\n??srail'in Muhafazakar Dostlar??\n_START_SECTION_\nFaaliyetleri\n_START_PARAGRAPH_\nGrubun 2005 stratejisi ile a??a????daki faaliyet alanlar?? tespit edilmi??tir:_NEWLINE_??srail'i destekleme",
                "bg": "\n_START_ARTICLE_\n?????????????????? ?? ???????????????? ??????????????????????\n_START_SECTION_\n???????????????????? ???? ??????????????????????????\n_START_PARAGRAPH_\n?? ?????????????????????? ???????? ???????? ???????????????????? ?? ???????????? ?????????????????????? ???? ?????????????????????? ?? ???????? ????????????",
                "ca": "\n_START_ARTICLE_\nAuchy-la-Montagne\n_START_SECTION_\nPoblaci??\n_START_PARAGRAPH_\nEl 2007 la poblaci?? de fet d'Auchy-la-Montagne era de 469 persones. Hi havia 160 fam??lies de les quals 28",
                "cs": "\n_START_ARTICLE_\n??emeslo\n_START_PARAGRAPH_\n??emeslo je ur??it?? druh manu??ln?? dovednosti, provozovan?? za ????elem ob??ivy, resp. vytv????en?? zisku. Pro ??emesln?? pr??ce je charakteristick?? vysok?? pod??l ru??n?? pr??ce, spojen?? s pou????v??n??m specializovan??ch n??stroj?? a pom??cek. ??emesln?? pr??ce",
                "da": "\n_START_ARTICLE_\n??ren??s slot\n_START_PARAGRAPH_\n??ren??s slot (svensk: ??ren??s slott) er et slot n??r Glumsl??v i Landskrona stad t??t p?? ??resunds-kysten i Sk??ne i Sverige._NEWLINE_??ren??s ligger",
                "el": "\n_START_ARTICLE_\n???????????? ??????????????\n_START_SECTION_\n???????????????????? ????????????????\n_START_PARAGRAPH_\n?? ???????????? ?????????????? ?????????????????? ???????? 17 ?????????????? 1976 ?????? ???????????????????? ?????? ?????????????????????? ??????",
                "et": "\n_START_ARTICLE_\nAus deutscher Geistesarbeit\n_START_PARAGRAPH_\nAus deutscher Geistesarbeit (alapealkiri Wochenblatt f??r wissenschaftliche und kulturelle Fragen der Gegenwart) oli ajakiri, mis 1924???1934 ilmus Tallinnas. Ajakirja andis 1932???1934",
                "fa": "\n_START_ARTICLE_\n?????????? ????????\n_START_PARAGRAPH_\n???????????? ???????? ???? ?????????? ???????? ??????. ?????? ???????? ?????????? ???? ???? ?????????? ?????????? ?? ???????????? ???? ?????????? ???????????? ???????????????? ???????? ?????????? ???????????????. ?????? ???????? ???? ?? ?????? ?????????? ?????????????",
                "fi": "\n_START_ARTICLE_\nBovesin veril??yly\n_START_SECTION_\nVeril??yly\n_START_PARAGRAPH_\n19. syyskuuta 1943 partisaaniryhm?? saapui Bovesiin tarkoituksenaan ostaa leip???? kyl??st??. Kyl??ss?? sattui olemaan kaksi SS-miest??, jotka",
                "he": "\n_START_ARTICLE_\n?????????? 85\n_START_SECTION_\n????????????????\n_START_PARAGRAPH_\n???????????? ?????????? ???????????? ?????????? ???????????? ?????????? ????????????. ?????????? ?????????????? ???????? ?????????? ?????????? ???????????? ???? ??????",
                "hi": "\n_START_ARTICLE_\n?????????\n_START_SECTION_\n????????? ??????????????????\n_START_PARAGRAPH_\n????????? ?????????????????? ?????? ????????????????????? ??????????????? 2007 ????????? ??????????????????????????? ??????????????? ??????????????? ?????????????????? ?????? ?????? ?????????????????? ?????? ????????? ????????? ?????? ?????? ????????? ?????????????????? ?????? ????????? 110",
                "hr": "\n_START_ARTICLE_\n??imariko (jezi??na porodica)\n_START_PARAGRAPH_\nChimarikan.-porodica sjevernoameri??kih indijanskih jezika koja prema Powersu obuhva??a jezike Indijanaca Chimariko (Chema??eko) sa rijeke Trinity i Chimalakwe",
                "hu": "\n_START_ARTICLE_\n??llami Politikai Igazgat??s??g\n_START_PARAGRAPH_\nAz ??llami Politikai Igazgat??s??g (r??vid??tve: GPU, oroszul: ?????????????????????????????? ???????????????????????? ????????????????????), majd k??s??bb Egyes??tett ??llami Politikai Igazgat??s??g Szovjet-Oroszorsz??g",
                "id": "\n_START_ARTICLE_\n(257195) 2008 QY41\n_START_SECTION_\nPembentukan\n_START_PARAGRAPH_\nSeperti asteroid secara keseluruhan, asteroid ini terbentuk dari nebula matahari primordial sebagai pecahan planetisimal, sesuatu di",
                "lt": "\n_START_ARTICLE_\n??avijos???Uardigo regionas\n_START_SECTION_\nGeografija\n_START_PARAGRAPH_\n??avijos-Uardigo regionas yra Atlanto vandenynu pakrant??s lygumoje",
                "lv": "\n_START_ARTICLE_\nApat??ts\n_START_SECTION_\n??pa????bas\n_START_PARAGRAPH_\nApat??ta kop??j?? ????misk?? formula ir Ca??????(PO???)???(OH,F,Cl)???, ir tr??s at????ir??gi apat??ta veidi: apat??ts: Ca??????(PO???)???(OH)???, fluorapat??ts Ca??????(PO???)???(F)??? un hlorapat??ts: Ca??????(PO???)???(Cl)???. P??c sast??va",
                "ms": "\n_START_ARTICLE_\nEdward C. Prescott\n_START_PARAGRAPH_\nEdward Christian Prescott (lahir 26 Disember 1940) ialah seorang ahli ekonomi Amerika. Beliau menerima Hadiah Peringatan Nobel dalam Sains Ekonomi pada tahun 2004, berkongsi",
                "no": "\n_START_ARTICLE_\nAl-Minya\n_START_SECTION_\nEtymologi\n_START_PARAGRAPH_\nDet er sprikende forklaringer p?? bynavnet. Det kan komme fra gammelegyptisk Men'at Khufu, i betydning byen hvor Khufu ble ammet, noe som knytter byen til farao Khufu (Keops), som",
                "ro": "\n_START_ARTICLE_\nDealurile Cern??u??iului\n_START_PARAGRAPH_\nDealurile Cern??u??iului sunt un lan?? deluros striat, care se ??ntinde ??n partea central?? a interfluviului dintre Prut ??i Siret, ??n cadrul regiunii Cern??u??i din",
                "sk": "\n_START_ARTICLE_\n10. peru?? RAAF\n_START_PARAGRAPH_\n10. peru?? RAAF je n??morn?? hliadkovacia peru?? kr????ovsk??ch austr??lskych vzdu??n??ch s??l (Royal Australian Air Force ??? RAAF) zalo??en?? na z??kladni Edinburgh v Ju??nej Austr??lii ako s????as?? 92",
                "sl": "\n_START_ARTICLE_\n105 Artemida\n_START_SECTION_\nOdkritje\n_START_PARAGRAPH_\nAsteroid je 16. septembra 1868 odkril James Craig Watson (1838 ??? 1880). Poimenovan je po Artemidi, boginji Lune iz gr??ke",
                "sr": "\n_START_ARTICLE_\n?????????? ?????????????? 1. ?????????????? (??????????????????????????)\n_START_SECTION_\n????????????????????????\n_START_PARAGRAPH_\n?????????? ???????????????? ???? 2010. ???????????? ?? ???????????? ???? ???????????? 212",
                "sv": "\n_START_ARTICLE_\n??stra Torps landskommun\n_START_SECTION_\nAdministrativ historik\n_START_PARAGRAPH_\nKommunen bildades i ??stra Torps socken i Vemmenh??gs h??rad i Sk??ne n??r 1862 ??rs kommunalf??rordningar tr??dde i kraft. _NEWLINE_Vid kommunreformen",
                "tl": "\n_START_ARTICLE_\nB??same Mucho\n_START_PARAGRAPH_\nAng B??same Mucho ay isang awit na nasa Kastila. Isinulat ito ng Mehikanang si Consuelo Vel??zquez noong 1940, bago sumapit ang kanyang ika-16 na",
                "uk": "\n_START_ARTICLE_\n?????????? ???? ???????? ??????????????\n_START_PARAGRAPH_\n???????????????? ?????????????????????????? ???????????????????? ?????????????? ?????????????????????????? ??????????????????, ?????????????????? ?????????? ?? ???????????????? ???????????????????? ???????????????? ?? ?????????? ?????????????????? ?? ???????????????????????????? ?????????? ??????????????. ?????????? ??????",
                "vi": "\n_START_ARTICLE_\n???????ng t???nh 316\n_START_PARAGRAPH_\n???????ng t???nh 316 hay t???nh l??? 316, vi???t t???t ??T316 hay TL316, l?? ???????ng t???nh ??? c??c huy???n Thanh S??n, Thanh Th???y, Tam N??ng t???nh Ph?? Th??? ._NEWLINE_??T316 b???t ?????u t??? x?? Tinh Nhu???",
                "multilingual-64k": "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada.",
                "multilingual-128k": "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada."}
#@title Enter your own seed (Optional).
user_seed = "???????? ???????????? ???????? ???? ???????? ???????????" #@param { type: "string" }
if user_seed.strip():
  seed = user_seed.strip()

# The seed must start with "_START_ARTICLE_" or the generated text will be gibberish
START_ARTICLE = "_START_ARTICLE_"
if START_ARTICLE not in seed:
  seed = "\n{}\n{}".format(START_ARTICLE, seed)

print("Generating text from seed:\n{}".format(seed))
#@title Initialize session.
with tf.Session(graph=g).as_default() as session:
  session.run(init_op)
#@title Generate text

with session.as_default():
  results = session.run([embeddings, neg_log_likelihood, ppl, activations, token_ids, generated_text], feed_dict={text: [seed]})
  embeddings_result, neg_log_likelihood_result, ppl_result, activations_result, token_ids_result, generated_text_result = results
  generated_text_output = generated_text_result[0].decode('utf-8')

print(generated_text_output)