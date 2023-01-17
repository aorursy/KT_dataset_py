# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import tensorflow as tf

import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_1d, global_max_pool

from tflearn.layers.merge_ops import merge

from tflearn.layers.estimator import regression

from tflearn.data_utils import to_categorical, pad_sequences



from tflearn.layers.core import activation





from keras.preprocessing.text import Tokenizer

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/winemag-data-130k-v2.csv')
train.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = train[train['variety'].notnull()]

import matplotlib as mpl

from pandas import DataFrame

DataFrame.hist(df, column='variety')
red_white_mapping = {"white": [ "Airén",

"Albariño",

"Aligoté",

"Arneis",

"Assyrtiko",

"Auxerrois",

"Colombard",

"Falanghina",

"Furmint",

"Friulano",

"Garganega",

"Glera",

"Godello",

"Greco",

"Grüner Veltliner",

"Inzolia",

"Kerner",

"Loureiro",

"Malvasia",

"Marsanne",

"Moschofilero",

"Müller-Thurgau",

"Palomino",

"Pecorino",

"Pedro Ximénez",

"Pinot blanc",

"Pinot gris/Pinot grigio",

"Piquepoul",

"Macabeo/Viura",

"Rhoditis",

"Ribolla Gialla",

"Rondinella",

"Roussanne",

"Scheurebe",

"Sémillon",

"Silvaner",

"Torrontés",

"Treixadura",

"Ugni blanc/Trebbiano",

"Verdejo",

"Verdicchio",

"Vermentino",

"Viognier",

"Welschriesling",

"Xarel·lo", "white blend",

                              

                              "addoraca",

"aidini",

"airén",

"alarije",

"albalonga",

"albana",

"albanella",

"albanello bianco",

"albaranzeuli bianco",

"albarello",

"albariño",

"albarola",

"albillo",

"aligoté",

"alionza",

"altesse",

"amigne",

"ansonica",

"antão vaz",

"arbane",

"arbois",

"arilla",

"arinto",

"arneis",

"arnsburger",

"arrufiac",

"arvesiniadu",

"asprinio bianco",

"assyrtiko",

"athiri",

"aubin blanc",

"aubin vert",

"aurore",

"auxerrois blanc",

"avesso",

"azal branco",

"bacchus",

"baco blanc",

"balzac blanc",

"baratuciat",

"barbera bianca",

"bariadorgia",

"baroque",

"bayan shirey",

"beba",

"besgano bianco",

"bia blanc",

"bianca",

"biancame",

"bianchetta trevigiana",

"bianco d'alessano",

"biancolella",

"biancone di portoferraio",

"bical",

"bigolona",

"blatterle",

"boal",

"bogdanuša",

"bombino bianco",

"borba blanca",

"bosco",

"bourboulenc",

"bouvier",

"breidecker",

"bronner",

"brustiano bianco",

"brustiano faux",

"bukettraube",

"burger",

"cabernet blanc",

"caíño blanco",

"camaralet",

"caprettone",

"carricante",

"cascarolo bianco",

"cassady",

"catarratto",

"cayetana",

"cayuga white",

"cereza",

"chambourcin",

"chardonel",

"chardonnay",

"chasan",

"chasselas",

"chenel",

"chenin blanc",

"chinuri",

"clairette blanche",

"claverie",

"cococciola",

"coda di pecora",

"coda di volpe",

"colombard",

"completer",

"cortese",

"courbu",

"crouchen",

"cserszegi fűszeres",

"debina",

"debit",

"diamond",

"dimiat",

"dinka",

"doña blanca",

"donzelinho branco",

"doradillo",

"drupeggio",

"durella",

"ehrenbreitsteiner",

"ehrenfelser",

"elbling",

"emir",

"encruzado",

"erbaluce",

"escanyavella",

"ezerjó",

"faberrebe",

"falanghina",

"favorita",

"fernão pires",

"fetească albă",

"fetească regală",

"fiano",

"findling",

"flora",

"folle blanche",

"forastera",

"forastera (spanish grape)",

"freisamer",

"furmint",

"gamay blanc gloriod",

"garganega",

"garrido fino",

"gewürztraminer",

"giró blanc",

"glera",

"godello",

"goldburger",

"goldriesling",

"gouais blanc",

"graisse",

"grasă de cotnari",

"grechetto",

"greco",

"green hungarian",

"grenache blanc",

"grillo",

"gringet",

"grk bijeli",

"gros manseng",

"grüner veltliner",

"gutenborner",

"hárslevelű",

"hebén",

"hibernal",

"hondarrabi zuri",

"humagne blanche",

"huxelrebe",

"incrocio manzoni 1.50",

"irsai olivér",

"jacquère",

"juhfark",

"juwel",

"kabar",

"kanzler",

"kéknyelű",

"kerner",

"knipperlé",

"koshu",

"kövérszőlő",

"krstač",

"l'acadie blanc",

"la crosse",

"lagarino bianco",

"lagorthi",

"lauzet",

"len de l'el",

"listán de huelva",

"loureira",

"luglienga",

"macabeo",

"maceratino",

"madeleine angevine",

"malagousia",

"malvar",

"malvasia",

"mantonico bianco",

"manzoni bianco",

"maraština",

"marawi",

"marsanne",

"marzemina bianca",

"mauzac",

"melody",

"melon de bourgogne",

"merlot blanc",

"merseguera",

"meslier-saint-françois",

"minella bianca",

"misket cherven",

"molette",

"mondeuse blanche",

"montepila",

"montonico bianco",

"montù",

"morio muscat",

"moscato giallo",

"moschofilero",

"mtsvane",

"müller-thurgau",

"muscadelle",

"muscat",

"muscat blanc à petits grains",

"muscat of alexandria",

"muscat ottonel",

"muscat d'eisenstadt",

"muscat rose à petits grains",

"muscat rouge à petits grains",

"narince",

"nasco",

"neuburger",

"neyret",

"noah",

"nobling",

"nosiola",

"nuragus",

"ondenc",

"optima",

"orléans",

"ortega",

"ortrugo",

"österreichisch-weiß",

"pallagrello bianco",

"palomino",

"pampanuto",

"pardillo",

"parellada",

"pascal blanc",

"passerina",

"pearl of csaba",

"pecorino",

"pedro giménez",

"pedro ximénez",

"perle",

"petit manseng",

"petit meslier",

"petite arvine",

"peurion",

"picardan",

"picolit",

"pigato",

"pinot blanc",

"pinot gris",

"piquepoul",

"plavay",

"pošip",

"prié blanc",

"prosecco",

"rabo de ovelha",

"räuschling",

"ravat blanc",

"regner",

"reichensteiner",

"retagliado bianco",

"rèze",

"rhoditis",

"ribolla gialla",

"rieslaner",

"riesling",

"rkatsiteli",

"robola",

"romorantin",

"rossese bianco",

"roter veltliner",

"rotgipfler",

"roublot",

"roupeiro",

"roussanne",

"rovello bianco",

"sacy",

"saint-pierre doré",

"sarfeher",

"sauvignon blanc",

"sauvignon gris",

"sauvignon vert",

"savagnin",

"savagnin rose",

"savatiano",

"scheurebe",

"sémillon",

"sercial",

"seyval blanc",

"siegerrebe",

"silvaner",

"smederevka",

"souvignier gris",

"st. pepin",

"sumoll",

"tamarez",

"taminga",

"tamjanika",

"tempranillo blanco",

"terrantez",

"terret blanc",

"terret gris",

"thrapsathiri",

"timorasso",

"torrontés",

"tourbat",

"traminette",

"trebbiano",

"treixadura",

"trousseau gris",

"tsolikouri",

"valvin muscat",

"vega",

"verdea",

"verdeca",

"verdejo",

"verdelho",

"verdello",

"verdesse",

"verdicchio",

"verdiso",

"verduzzo",

"verduzzo trevigiano",

"vermentino",

"vernaccia",

"vernaccia di oristano",

"versoaln",

"vespaiola",

"vidal blanc",

"vigiriega",

"vignoles",

"vilana",

"viognier",

"viosinho",

"vital",

"vitovska",

"vugava",

"weldra",

"welschriesling",

"swish beverages",

"würzer",

"xarel·lo",

"xynisteri",

"zalema",

"zéta",

"zierfandler",

"žilavka"

],



"red": [

"Agiorgitiko",

"Aglianico",

"Baco noir",

"Barbera",

"Blauburger",

"Blaufränkisch",

"Bobal",

"Brachetto",

"Carignan",

"Carmenère",

"Cesanese Comune",

"Chambourcin",

"Chasselas",

"Cinsaut",

"Corvina",

"Dolcetto",

"Douce noir/Charbono/Bonarda",

"Frappato",

"Gamay",

"Grenache/Garnacha",

"Gaglioppo",

"Graciano",

"Gros Manseng",

"Lagrein",

"Lambrusco",

"Malbec",

"Mencía",

"Montepulciano",

"Mourvèdre/Monastrell/Mataro",

"Nebbiolo",

"Negroamaro",

"Négrette",

"Nero d'Avola",

"Nerello",

"Petite sirah/Durif",

"Petit verdot",

"Pinot Meunier",

"Pinotage",

"Poulsard",

"Ruché",

"Sagrantino",

"Sangiovese",

"Schiava",

"St. Laurent",

"Tannat",

"Tempranillo",

"Tibouren",

"Touriga Nacional",

"Trepat",

"Trousseau",

"Uhudler",

"Xinomavro",

"Zinfandel/Primitivo",

"Zweigelt", "red blend",

"abbuoto",

"abouriou",

"abrusco",

"acitana",

"acolon",

"adakarası",

"agh shani",

"agiorgitiko",

"aglianico",

"aglianicone",

"albaranzeuli nero",

"albarossa",

"aleatico",

"aleksandrouli",

"alfrocheiro preto",

"alicante bouschet",

"alicante ganzin",

"alvarelhão",

"ancellotta",

"aramon",

"argaman",

"argant",

"arrouya noir",

"aspiran",

"aubun",

"avanà",

"avarengo",

"azal tinto",

"băbească neagră",

"babić",

"bachet noir",

"baco noir",

"baga",

"barbarossa",

"barbaroux",

"barbera",

"barbera del sannio",

"barbera sarda",

"barsaglina",

"beaunoir",

"bellone",

"béquignol noir",

"black muscat",

"blatina",

"blauburger",

"blauer portugieser",

"blaufränkisch",

"bobal",

"boğazkere",

"bombino nero",

"bonamico",

"bonarda piemontese",

"bonda",

"bondola",

"bouchalès",

"bouteillan noir",

"bovale",

"bracciola nera",

"brachetto",

"braquet",

"brugnola",

"brun argenté",

"brun fourca",

"bubbierasco",

"busuioacă de bohotin",

"cabernet dorsa",

"cabernet franc",

"cabernet gernischt",

"cabernet mitos",

"cabernet sauvignon",

"caiño tinto",

"calabrese montenuovo",

"caladoc",

"calitor",

"çalkarası",

"camaraou noir",

"canaiolo",

"canari noir",

"carignan",

"carménère",

"cascade",

"castelão",

"castets",

"catanese nero",

"catawba",

"cesanese comune",

"césar",

"chancellor",

"chatus (wine grape)",

"chelois",

"cienna",

"ciliegiolo",

"cinsaut",

"clinton",

"colombana nera",

"colorino",

"complexa",

"cornalin d'aoste",

"cornifesto",

"corot noir",

"corvina",

"corvinone",

"couderc noir",

"counoise",

"criolla grande",

"croatina",

"cygne blanc",

"dameron",

"de chaunac",

"delaware",

"diolinoir",

"dobričić",

"dolcetto",

"domina",

"dornfelder",

"douce noir",

"douce noire grise",

"drnekuša",

"dunkelfelder",

"duras",

"dureza",

"durif",

"ederena",

"emperor",

"enfariné noir",

"espadeiro",

"étraire de la dui",

"fer",

"fetească neagră",

"flora",

"flot rouge",

"forcallat tinta",

"fortana",

"frappato",

"freisa",

"frontenac",

"frühroter veltliner",

"fuella",

"fumin",

"gaglioppo",

"gamaret",

"gamay",

"gamay beaujolais",

"garanoir",

"garró",

"girò",

"gouget noir",

"graciano",

"grand noir de la calmette",

"grenache",

"grignolino",

"grisa nera",

"grolleau",

"groppello",

"gros verdot",

"gueuche noir",

"helfensteiner",

"heroldrebe",

"hondarribi beltza",

"hron",

"incrocio manzoni 2.14",

"incrocio manzoni 2.15",

"isabella",

"ives noir",

"jaén tinto",

"joubertin",

"juan garcía",

"kadarka",

"kalecik karası",

"kotsifali",

"krasnostop zolotovsky",

"kratosija",

"lacrima",

"lagrein",

"lambrusco",

"landal noir",

"landot noir",

"léon millot",

"liatiko",

"limnio",

"listán negro",

"madrasa",

"magarach ruby",

"magliocco canino",

"magliocco dolce",

"malbec",

"mammolo",

"mandilaria",

"manseng noir",

"manto negro",

"maratheftiko",

"marechal foch",

"marechal joffre",

"marselan",

"marzemino",

"mauzac noir",

"mavro",

"mavrodafni",

"mavrud",

    "merille",

"merlot",

"milgranet",

"mission",

"molinara",

"mondeuse noire",

"monica",

"montepulciano",

"montepulciano d'abruzzo",

"montù",

"moreto",

"moristel",

"mornen noir",

"morrastel bouschet",

"mourisco tinto",

"mourvèdre",

"mtevandidi",

"mujuretuli",

"mureto",

"muscardin",

"muscat bleu",

"nebbiolo",

"negoska",

"negrara",

"négrette",

"negroamaro",

"nerello",

"nero d'avola",

"nielluccio",

"nocera",

"noiret",

"norton",

"oeillade noire",

"öküzgözü",

"pais",

"pallagrello nero",

"pamid",

"papazkarası",

"parraleta",

"pascale di cagliari",

"pelaverga",

"peloursin",

"perricone",

"persan",

"petit bouschet",

"petit rouge",

"petit verdot",

"piccola nera",

"piedirosso",

"pignolo",

"pineau d'aunis",

"pinot meunier",

"pinot noir",

"pinot noir précoce",

"pinotage",

"pione",

"piquepoul",

"plantet",

"plassa",

"plavac mali",

"pollera nera",

"portan",

"poulsard",

"prieto picudo",

"prokupac",

"prunesta",

"raboso",

"ramisco",

"refosco",

"refosco dal peduncolo rosso",

"rimava",

"roesler",

"romé",

"romeiko",

"rondinella",

"rosette",

"rossignola",

"rossola nera",

"rossolino nero",

"rotberger",

"rouge du pays",

"royalty",

"ruby cabernet",

"ruché",

"rufete",

"sagrantino",

"salvador",

"san giuseppe nero",

"sangiovese",

"saperavi",

"schioppettino",

"schönburger",

"sciacarello",

"sciascinoso",

"ségalin",

"servanin",

"severny",

"seyval noir",

"siroka melniska",

"sjriak",

"sousao",

"sousão",

"souson",

"sousón",

"souzão",

"st. laurent",

"stanušina crna",

"sumoll",

"susac crni",

"susumaniello",

"swenson red",

"syrah",

"taferielt",

"tannat",

"tarrango",

"tazzelenghe",

"teinturier",

"tempranillo",

"téoulier",

"termarina rossa",

"teroldego",

"terrano",

"terret noir",

"tibouren",

"tinta amarela",

"tinta barroca",

"tinta cão",

"tinta carvalha",

"tinta francisca",

"tinta miuda",

"tinta negra mole",

"touriga francesa",

"touriga nacional",

"trepat",

"tressot",

"trevisana nera",

"triomphe d'alsace",

"trollinger",

"trousseau",

"tsardana",

"uva di troia",

"uva rara",

"uva tosca",

"uvalino",

"valdiguié",

"valentino nero",

"vermentino nero",

"vespolina",

"vien de nus",

"vitis rotundifolia",

"vranac",

"vuillermin",

"wildbacher",

"xinomavro",

"žametovka",

"zarya severa",

"zinfandel",

"zweigelt"





]}
inverted_mapping = {}

for k,arr in red_white_mapping.items():

    for v in arr:

        if v.lower() not in inverted_mapping:

            inverted_mapping[v.lower()] = k
varieties = df['variety'].unique()
mapped = [inverted_mapping[variety.lower()] if variety.lower() in inverted_mapping else variety.lower() for variety in varieties ]
unmappable = [variety for variety in mapped if variety is not 'red' and variety is not 'white']
print(len(unmappable))

print(unmappable)
# reduce to only the known varieties

trainDF = df[df.apply(lambda x: x['variety'].lower() in inverted_mapping, axis=1, reduce=True)]

# add color

trainDF['color'] = trainDF.apply (lambda row: inverted_mapping[row['variety'].lower()] ,axis=1)

num_varieties = len(trainDF['color'].unique())

print(num_varieties)

encoder = LabelEncoder()

onehot = OneHotEncoder(num_varieties, sparse=False)

tokenizer = Tokenizer(filters='—–!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
# train tokenizer

tokenizer.fit_on_texts(trainDF.description.values)



# tokenize

words = tokenizer.texts_to_sequences(trainDF.description.values)



sequence_len = max(map(len, words))

# pad

trainX = pad_sequences(list(words),maxlen=sequence_len, value=0.)



varieties = np.array(trainDF.color.values)

# fit the encoders

varieties = encoder.fit_transform(varieties)

onehot.fit(varieties.reshape(varieties.size,1))

#transform data with encoders

trainY = encoder.transform(trainDF.color.values)

trainY = onehot.transform(trainY.reshape(trainY.size,1))

print(trainY.shape)
print(trainX.shape, trainY.shape)
from keras.layers import Dense, Input, Flatten, Dropout

from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, Activation, Dense

from keras.models import Sequential



model = Sequential()

model.add(Embedding(len(tokenizer.word_index), 512, input_length=sequence_len))



#model.add(Conv1D(3, 3, padding='same', input_shape=(sequence_len,), activation='relu'))

#model.add(Dropout(0.2))



model.add(Conv1D(3, 2, padding='same', activation='relu'))

model.add(Dropout(0.2))



model.add(Conv1D(3, 1, padding='same', activation='relu'))

model.add(Dropout(0.2))



model.add(GlobalMaxPooling1D())

#model.add(Flatten())

model.add(Dense(200, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_varieties, activation='relu'))





model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(trainX, trainY, validation_split=0.1, epochs=10)