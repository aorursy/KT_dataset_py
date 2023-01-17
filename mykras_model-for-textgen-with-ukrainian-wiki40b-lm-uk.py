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
                "ar": "\n_START_ARTICLE_\nأوليفيا كوك\n_START_SECTION_\nنشأتها والتعلي \n_START_PARAGRAPH_\nولدت أوليفيا كوك في أولدهام في مانشستر الكبرى لأسرة تتكون من أب يعمل كظابط شرطة، وأمها تعمل كممثلة مبيعات. عندما كانت صغيرة بدأت تأخذ دروساً في الباليه الجمباز. وفي المدرسة شاركت في المسرحيات المدرسية، إضافةً إلى عملها في مسرح سندريلا . وفي سن الرابعة عشر عاماً، حصلت على وكيلة لها في مانشستر وهي وقعت عقداً مع وكالة الفنانين المبدعين في مانشستر،",
                "zh-cn": "\n_START_ARTICLE_\n上尾事件\n_START_SECTION_\n日本国铁劳资关系恶化\n_START_PARAGRAPH_\n由于日本国铁财政恶化，管理层开始重整人手安排，令工会及员工感到受威胁。但日本国铁作为公营企业，其雇员均受公营企业等劳资关系法规管——该法第17条规定公营企业员工不得发动任何罢工行为。为了规避该法例",
                "zh-tw": "\n_START_ARTICLE_\n乌森\n_START_PARAGRAPH_\n烏森（法語：Houssen，發音：[usən]；德語：Hausen；阿爾薩斯語：Hüse）是法國上萊茵省的一個市鎮，位於該省北部，屬於科爾馬-里博維萊區（Colmar-Ribeauvillé）第二科爾馬縣（Colmar-2）。該市鎮總面積6.7平方公里，2009年時的人口為",
                "nl": "\n_START_ARTICLE_\n1001 vrouwen uit de Nederlandse geschiedenis\n_START_SECTION_\nSelectie van vrouwen\n_START_PARAGRAPH_\nDe 'oudste' biografie in het boek is gewijd aan de beschermheilige",
                "fr": "\n_START_ARTICLE_\nꝹ\n_START_SECTION_\nUtilisation\n_START_PARAGRAPH_\nLe d insulaire est utilisé comme lettre additionnelle dans l’édition de 1941 du recueil de chroniques galloises Brut y Tywysogion",
                "de": "\n_START_ARTICLE_\nÜnal Demirkıran\n_START_SECTION_\nLaufbahn\n_START_PARAGRAPH_\nDemirkıran debütierte als junges Talent am 25. September 1999 im Auswärtsspiel des SSV Ulm 1846 bei Werder Bremen (2:2) in der Bundesliga, als er kurz",
                "it": "\n_START_ARTICLE_\n28th Street (linea IRT Lexington Avenue)\n_START_SECTION_\nStoria\n_START_PARAGRAPH_\nLa stazione, i cui lavori di costruzione ebbero inizio nel 1900, venne aperta il 27 ottobre 1904, come",
                "ja": "\n_START_ARTICLE_\nしのぶ・まさみshow'05 恋してラララ\n_START_SECTION_\n概要\n_START_PARAGRAPH_\n『上海ルーキーSHOW』の打ち切り後に放送された年末特番で、同番組MCの大竹しのぶと久本雅美が恋愛にまつわるテーマでトークや音楽企画を展開していた。基本は女",
                "ko": "\n_START_ARTICLE_\n녹턴, Op. 9 (쇼팽)\n_START_SECTION_\n녹턴 3번 나장조\n_START_PARAGRAPH_\n쇼팽의 녹턴 3번은 세도막 형식인 (A-B-A)형식을 취하고 있다. 첫 부분은 알레그레토(Allegretto)의 빠르기가 지시되어 있으며 물 흐르듯이 부드럽게 전개되나",
                "pl": "\n_START_ARTICLE_\nAK-176\n_START_SECTION_\nHistoria\n_START_PARAGRAPH_\nPod koniec lat 60 XX w. w ZSRR dostrzeżono potrzebę posiadania lekkiej armaty uniwersalnej średniego kalibru o stosunkowo dużej mocy ogniowej, która",
                "pt": "\n_START_ARTICLE_\nÁcido ribonucleico\n_START_SECTION_\nIntermediário da transferência de informação\n_START_PARAGRAPH_\nEm 1957 Elliot Volkin e Lawrence Astrachan fizeram uma observação significativa. Eles descobriram que uma das mais marcantes mudanças",
                "ru": "\n_START_ARTICLE_\nАрнольд, Ремо\n_START_SECTION_\nКлубная карьера\n_START_PARAGRAPH_\nАрнольд перешёл в академию «Люцерна» в 12 лет. С 2014 года выступал за вторую команду, где провёл пятнадцать встреч. С сезона 2015/2016 находится в составе основной команды. 27 сентября 2015 года дебютировал",
                "es": "\n_START_ARTICLE_\n(200012) 2007 LK20\n_START_SECTION_\nDesignación y nombre\n_START_PARAGRAPH_\nDesignado provisionalmente como 2007 LK20.\n_START_SECTION_\nCaracterísticas orbitales\n_START_PARAGRAPH_\n2007 LK20",
                "th": "\n_START_ARTICLE_\nการนัดหยุดเรียนเพื่อภูมิอากาศ\n_START_SECTION_\nเกรียตา ทืนแบร์ย\n_START_PARAGRAPH_\nวันที่ 20 สิงหาคม 2561 เกรียตา ทืนแบร์ย นักกิจกรรมภูมิอากาศชาวสวีเดน ซึ่งขณะนั้นศึกษาอยู่ในชั้นเกรด 9 (เทียบเท่ามัธยมศึกษาปีที่ 3) ตัดสินใจไม่เข้าเรียนจนกระทั่งการเลือกตั้งทั่วไปในประเทศสวีเดนปี",
                "tr": "\n_START_ARTICLE_\nİsrail'in Muhafazakar Dostları\n_START_SECTION_\nFaaliyetleri\n_START_PARAGRAPH_\nGrubun 2005 stratejisi ile aşağıdaki faaliyet alanları tespit edilmiştir:_NEWLINE_İsrail'i destekleme",
                "bg": "\n_START_ARTICLE_\nАвтомобил с повишена проходимост\n_START_SECTION_\nОсобености на конструкцията\n_START_PARAGRAPH_\nВ исторически план леки автомобили с висока проходимост се произвеждат и имат военно",
                "ca": "\n_START_ARTICLE_\nAuchy-la-Montagne\n_START_SECTION_\nPoblació\n_START_PARAGRAPH_\nEl 2007 la població de fet d'Auchy-la-Montagne era de 469 persones. Hi havia 160 famílies de les quals 28",
                "cs": "\n_START_ARTICLE_\nŘemeslo\n_START_PARAGRAPH_\nŘemeslo je určitý druh manuální dovednosti, provozovaný za účelem obživy, resp. vytváření zisku. Pro řemeslné práce je charakteristický vysoký podíl ruční práce, spojený s používáním specializovaných nástrojů a pomůcek. Řemeslné práce",
                "da": "\n_START_ARTICLE_\nÖrenäs slot\n_START_PARAGRAPH_\nÖrenäs slot (svensk: Örenäs slott) er et slot nær Glumslöv i Landskrona stad tæt på Øresunds-kysten i Skåne i Sverige._NEWLINE_Örenäs ligger",
                "el": "\n_START_ARTICLE_\nΆλβαρο Ρεκόμπα\n_START_SECTION_\nΒιογραφικά στοιχεία\n_START_PARAGRAPH_\nΟ Άλβαρο Ρεκόμπα γεννήθηκε στις 17 Μαρτίου 1976 στο Μοντεβίδεο της Ουρουγουάης από",
                "et": "\n_START_ARTICLE_\nAus deutscher Geistesarbeit\n_START_PARAGRAPH_\nAus deutscher Geistesarbeit (alapealkiri Wochenblatt für wissenschaftliche und kulturelle Fragen der Gegenwart) oli ajakiri, mis 1924–1934 ilmus Tallinnas. Ajakirja andis 1932–1934",
                "fa": "\n_START_ARTICLE_\nتفسیر بغوی\n_START_PARAGRAPH_\nایرانی حسین بن مسعود بغوی است. این کتاب خلاصه ای از تفسیر الکشف و البیان عن تفسیر القرآن ابواسحاق احمد ثعلبی می‌باشد. این کتاب در ۴ جلد موجود می‌باش",
                "fi": "\n_START_ARTICLE_\nBovesin verilöyly\n_START_SECTION_\nVerilöyly\n_START_PARAGRAPH_\n19. syyskuuta 1943 partisaaniryhmä saapui Bovesiin tarkoituksenaan ostaa leipää kylästä. Kylässä sattui olemaan kaksi SS-miestä, jotka",
                "he": "\n_START_ARTICLE_\nאוגדה 85\n_START_SECTION_\nהיסטוריה\n_START_PARAGRAPH_\nהאוגדה הוקמה בהתחלה כמשלט העמקים בשנות השבעים. בשנות השמונים הפכה להיות אוגדה מרחבית עם שתי",
                "hi": "\n_START_ARTICLE_\nऑडी\n_START_SECTION_\nऑडी इंडिया\n_START_PARAGRAPH_\nऑडी इंडिया की स्थापना मार्च 2007 में फोक्सवैगन ग्रुप सेल्स इंडिया के एक विभाजन के रूप में की गई थी। दुनिया भर में 110",
                "hr": "\n_START_ARTICLE_\nČimariko (jezična porodica)\n_START_PARAGRAPH_\nChimarikan.-porodica sjevernoameričkih indijanskih jezika koja prema Powersu obuhvaća jezike Indijanaca Chimariko (Chemaŕeko) sa rijeke Trinity i Chimalakwe",
                "hu": "\n_START_ARTICLE_\nÁllami Politikai Igazgatóság\n_START_PARAGRAPH_\nAz Állami Politikai Igazgatóság (rövidítve: GPU, oroszul: Государственное политическое управление), majd később Egyesített Állami Politikai Igazgatóság Szovjet-Oroszország",
                "id": "\n_START_ARTICLE_\n(257195) 2008 QY41\n_START_SECTION_\nPembentukan\n_START_PARAGRAPH_\nSeperti asteroid secara keseluruhan, asteroid ini terbentuk dari nebula matahari primordial sebagai pecahan planetisimal, sesuatu di",
                "lt": "\n_START_ARTICLE_\nŠavijos–Uardigo regionas\n_START_SECTION_\nGeografija\n_START_PARAGRAPH_\nŠavijos-Uardigo regionas yra Atlanto vandenynu pakrantės lygumoje",
                "lv": "\n_START_ARTICLE_\nApatīts\n_START_SECTION_\nĪpašības\n_START_PARAGRAPH_\nApatīta kopējā ķīmiskā formula ir Ca₁₀(PO₄)₆(OH,F,Cl)₂, ir trīs atšķirīgi apatīta veidi: apatīts: Ca₁₀(PO₄)₆(OH)₂, fluorapatīts Ca₁₀(PO₄)₆(F)₂ un hlorapatīts: Ca₁₀(PO₄)₆(Cl)₂. Pēc sastāva",
                "ms": "\n_START_ARTICLE_\nEdward C. Prescott\n_START_PARAGRAPH_\nEdward Christian Prescott (lahir 26 Disember 1940) ialah seorang ahli ekonomi Amerika. Beliau menerima Hadiah Peringatan Nobel dalam Sains Ekonomi pada tahun 2004, berkongsi",
                "no": "\n_START_ARTICLE_\nAl-Minya\n_START_SECTION_\nEtymologi\n_START_PARAGRAPH_\nDet er sprikende forklaringer på bynavnet. Det kan komme fra gammelegyptisk Men'at Khufu, i betydning byen hvor Khufu ble ammet, noe som knytter byen til farao Khufu (Keops), som",
                "ro": "\n_START_ARTICLE_\nDealurile Cernăuțiului\n_START_PARAGRAPH_\nDealurile Cernăuțiului sunt un lanț deluros striat, care se întinde în partea centrală a interfluviului dintre Prut și Siret, în cadrul regiunii Cernăuți din",
                "sk": "\n_START_ARTICLE_\n10. peruť RAAF\n_START_PARAGRAPH_\n10. peruť RAAF je námorná hliadkovacia peruť kráľovských austrálskych vzdušných síl (Royal Australian Air Force – RAAF) založená na základni Edinburgh v Južnej Austrálii ako súčasť 92",
                "sl": "\n_START_ARTICLE_\n105 Artemida\n_START_SECTION_\nOdkritje\n_START_PARAGRAPH_\nAsteroid je 16. septembra 1868 odkril James Craig Watson (1838 – 1880). Poimenovan je po Artemidi, boginji Lune iz grške",
                "sr": "\n_START_ARTICLE_\nЉанос Морелос 1. Сексион (Истапангахоја)\n_START_SECTION_\nСтановништво\n_START_PARAGRAPH_\nПрема подацима из 2010. године у насељу је живело 212",
                "sv": "\n_START_ARTICLE_\nÖstra Torps landskommun\n_START_SECTION_\nAdministrativ historik\n_START_PARAGRAPH_\nKommunen bildades i Östra Torps socken i Vemmenhögs härad i Skåne när 1862 års kommunalförordningar trädde i kraft. _NEWLINE_Vid kommunreformen",
                "tl": "\n_START_ARTICLE_\nBésame Mucho\n_START_PARAGRAPH_\nAng Bésame Mucho ay isang awit na nasa Kastila. Isinulat ito ng Mehikanang si Consuelo Velázquez noong 1940, bago sumapit ang kanyang ika-16 na",
                "uk": "\n_START_ARTICLE_\nІслам та інші релігії\n_START_PARAGRAPH_\nПротягом багатовікової ісламської історії мусульманські правителі, ісламські вчені і звичайні мусульмани вступали у різні відносини з представниками інших релігій. Стиль цих",
                "vi": "\n_START_ARTICLE_\nĐường tỉnh 316\n_START_PARAGRAPH_\nĐường tỉnh 316 hay tỉnh lộ 316, viết tắt ĐT316 hay TL316, là đường tỉnh ở các huyện Thanh Sơn, Thanh Thủy, Tam Nông tỉnh Phú Thọ ._NEWLINE_ĐT316 bắt đầu từ xã Tinh Nhuệ",
                "multilingual-64k": "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada.",
                "multilingual-128k": "\n_START_ARTICLE_\n1882 Prince Edward Island general election\n_START_PARAGRAPH_\nThe 1882 Prince Edward Island election was held on May 8, 1882 to elect members of the House of Assembly of the province of Prince Edward Island, Canada."}
#@title Enter your own seed (Optional).
user_seed = "Хіба ревуть воли як ясла повні?" #@param { type: "string" }
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