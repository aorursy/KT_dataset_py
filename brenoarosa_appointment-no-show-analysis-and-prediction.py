import re

import unicodedata

import io



import joblib



import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier



import xgboost as xgb

from xgboost.sklearn import XGBClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
df = pd.read_csv("../input/KaggleV2-May-2016.csv", index_col=None,

                 parse_dates=["ScheduledDay", "AppointmentDay"], infer_datetime_format=True)



def camelcase_to_snakecase(name):

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)

    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()



df.columns = df.columns.map(camelcase_to_snakecase)

df.rename(columns={'hipertension':'hypertension', 'handcap':'handicap'}, inplace=True)



df.sort_values(["scheduled_day","appointment_day"], inplace=True, ascending=True)  
df.head()
mean_csv = io.StringIO(""""Mesorregiões, microrregiões, municípios, distritos, subdistritos e bairros", "Valor do rendimento nominal mediano mensal das pessoas de 10 anos ou mais de idade (R$)" 

"Andorinhas",510.00

"Antonio Honório",755.00

"Ariovaldo Favalessa",510.00

"Barro Vermelho",2000.00

"Bela Vista",510.00

"Bento Ferreira",1500.00

"Boa Vista",590.00

"Bonfim",510.00

"Carapina I",700.00

"Caratoíra",510.00

"Centro",1000.00

"Comdusa",510.00

"Conquista",510.00

"Consolação",510.00

"Cruzamento",510.00

"da Penha",510.00

"De Fátima",795.00

"de Lourdes",1000.00

"do Cabral",510.00

"do Moscoso",510.00

"do Quadro",510.00

"Enseada do Suá",2000.00

"Estrelinha",510.00

"Fonte Grande",510.00

"Forte São João",510.00

"Fradinhos",1000.00

"Goiabeiras",550.00

"Grande Vitória",510.00

"Gurigica",510.00

"Hélio Ferraz",600.00

"Horto",510.00

"Ilha das Caieiras",510.00

"Ilha de Santa Maria",560.00

"Ilha do Boi",1500.00

"Ilha do Frade",2500.00

"Ilha do Príncipe",510.00

"Inhanguetá",510.00

"Itararé",510.00

"Jabour",1000.00

"Jardim Camburí",1500.00

"Jardim da Penha",1500.00

"Jesus de Nazareth",510.00

"Joana D'arc",510.00

"Jucutuquara",800.00

"Maria Ortiz",510.00

"Mário Cypreste",510.00

"Maruípe",800.00

"Mata da Praia",2000.00

"Monte Belo",511.00

"Morada de Camburí",1100.00

"Nazareth",850.00

"Nova Palestina",510.00

"Parque Industrial",405.00

"Parque Moscoso",1005.00

"Piedade",510.00

"Pontal de Camburí",800.00

"Praia do Canto",2000.00

"Praia do Suá",510.00

"Redenção",510.00

"República",900.00

"Resistência",510.00

"Romão",510.00

"Santa Cecília",1000.00

"Santa Clara",700.00

"Santa Helena",2500.00

"Santa Lúcia",1500.00

"Santa Luíza",1000.00

"Santa Martha",510.00

"Santa Tereza",510.00

"Santo André",510.00

"Santo Antônio",510.00

"Santos Dumont",510.00

"Santos Reis",510.00

"São Benedito",440.00

"São Cristovão",510.00

"São José",510.00

"São Pedro",510.00

"Segurança do Lar",700.00

"Solon Borges",700.00

"Tabuazeiro",510.00

"Universitário",600.00

"Vila Rubim",510.00""")
# This data contains the mean incoming of a given neighbourhood.



mean_income = pd.read_csv(mean_csv, index_col=None)

mean_income.rename(columns={mean_income.columns[0]: 'neighbourhood',

                            mean_income.columns[1]: 'neigh_mean_income'},

                    inplace=True)
mean_income.head()
ranges_csv = io.StringIO(""""Mesorregiões, microrregiões, municípios, distritos, subdistritos e bairros","Até ½ salário mínimo","Mais de 1/2 a 1 salário mínimo","Mais de 1 a 2 salário mínimo","Mais de 2 a 5 salário mínimo","Mais de 5 a 10 salário mínimo","Mais de 10 a 20 salário mínimo","Mais de 20 salário mínimo","Sem rendimento (2)"

"Andorinhas",0.024348810872027,0.147791619479049,0.293318233295583,0.148924122310306,0.019818799546999,0.002831257078143,0,0.362967157417894

"Antonio Honório",0.012841091492777,0.123595505617978,0.177367576243981,0.219903691813804,0.12199036918138,0.035313001605136,0.01123595505618,0.297752808988764

"Ariovaldo Favalessa",0.030927835051546,0.185567010309278,0.241531664212077,0.163475699558174,0.044182621502209,0.007363770250368,0,0.326951399116348

"Barro Vermelho",0.003528719858851,0.034503038619879,0.065281317388747,0.162713193491472,0.201137031954519,0.173887473044501,0.080572436777103,0.278376788864928

"Bela Vista",0.038157536113382,0.20087217225402,0.266830198964295,0.118560915780867,0.022349414009267,0.002725538293813,0.000272553829381,0.350231670754974

"Bento Ferreira",0.004942665085014,0.065243179122183,0.104389086595492,0.217477263740609,0.205417160933175,0.111704230921313,0.029458283906683,0.261368129695532

"Boa Vista",0.016472868217054,0.169573643410853,0.27422480620155,0.163759689922481,0.068798449612403,0.02422480620155,0.002906976744186,0.280038759689922

"Bonfim",0.034505640345056,0.209024552090245,0.250165892501659,0.136031851360319,0.031851360318514,0.00464499004645,0.001658925016589,0.332116788321168

"Carapina I",0.013577023498695,0.123237597911227,0.213577023498694,0.256396866840731,0.084073107049608,0.019321148825065,0.002610966057441,0.287206266318538

"Caratoíra",0.02876557191393,0.213363533408834,0.226047565118913,0.098301245753114,0.020158550396376,0.003850509626274,0.000453001132503,0.409060022650057

"Centro",0.00905977240084,0.096895370677273,0.166611424152027,0.255441387691968,0.153795160755718,0.053916694287924,0.01281626339631,0.25146392663794

"Comdusa",0.012113055181696,0.231493943472409,0.238223418573351,0.045760430686407,0.004037685060565,0,0,0.468371467025572

"Conquista",0.062087186261559,0.252972258916777,0.229854689564069,0.064729194187583,0.015852047556143,0.003963011889036,0.000660501981506,0.369881109643329

"Consolação",0.016427969671441,0.174389216512216,0.209351305812974,0.13647851727043,0.057287278854254,0.011794439764111,0.002948609941028,0.391322662173547

"Cruzamento",0.025012761613068,0.25114854517611,0.23226135783563,0.081674323634507,0.02246043899949,0.00357325165901,0.001020929045431,0.382848392036753

"da Penha",0.025201185938162,0.223845828038967,0.224481152054214,0.115840745446845,0.029648454044896,0.003176620076239,0.000423549343499,0.377382465057179

"De Fátima",0.01078914919852,0.105425400739827,0.188964241676942,0.240135635018496,0.116214549938348,0.028668310727497,0.004932182490752,0.304870530209618

"de Lourdes",0.006916426512968,0.092795389048991,0.152737752161383,0.237463976945245,0.157348703170029,0.061671469740634,0.021902017291066,0.269164265129683

"do Cabral",0.034609720176731,0.258468335787923,0.221649484536082,0.102356406480118,0.019145802650957,0.005154639175258,0,0.358615611192931

"do Moscoso",0.035060975609756,0.245426829268293,0.214939024390244,0.11280487804878,0.032012195121951,0.004573170731707,0.001524390243902,0.353658536585366

"do Quadro",0.046103183315038,0.170142700329308,0.249176728869374,0.153677277716795,0.052689352360044,0.010976948408343,0.001097694840834,0.316136114160263

"Enseada do Suá",0.008583690987124,0.065450643776824,0.080472103004292,0.167381974248927,0.218884120171674,0.13519313304721,0.064377682403434,0.259656652360515

"Estrelinha",0.047810770005033,0.21389028686462,0.269250125817816,0.105686965274283,0.016607951685959,0.002013085052843,0.000503271263211,0.344237544036236

"Fonte Grande",0.041860465116279,0.26046511627907,0.228837209302326,0.091162790697674,0.032558139534884,0.009302325581395,0.004651162790698,0.331162790697674

"Forte São João",0.025506376594149,0.225806451612903,0.258064516129032,0.118529632408102,0.033008252063016,0.012003000750188,0.00450112528132,0.32258064516129

"Fradinhos",0.005576208178439,0.105328376703841,0.144361833952912,0.208798017348203,0.146220570012392,0.075588599752169,0.026641883519207,0.287484510532838

"Goiabeiras",0.015578947368421,0.168,0.236631578947368,0.182315789473684,0.065684210526316,0.017684210526316,0.003368421052632,0.310736842105263

"Grande Vitória",0.0366391184573,0.216804407713499,0.249035812672176,0.088154269972452,0.015151515151515,0.002479338842975,0.000550964187328,0.391184573002755

"Gurigica",0.042451894465384,0.242610593136282,0.250148780003967,0.085300535608014,0.012497520333267,0.002182106724856,0.000396746677247,0.364411823050982

"Hélio Ferraz",0.016167192429022,0.134069400630915,0.242902208201893,0.204258675078864,0.079652996845426,0.014984227129338,0.001577287066246,0.306388012618297

"Horto",0.044943820224719,0.168539325842697,0.146067415730337,0.146067415730337,0.112359550561798,0.02247191011236,0,0.359550561797753

"Ilha das Caieiras",0.032152230971129,0.245406824146982,0.236220472440945,0.088582677165354,0.013779527559055,0.001312335958005,0,0.38254593175853

"Ilha de Santa Maria",0.020280057943023,0.188314823756639,0.259777885079672,0.188797682279092,0.061323032351521,0.011105746016417,0.002414292612265,0.267986479961371

"Ilha do Boi",0.011764705882353,0.054901960784314,0.088235294117647,0.109803921568627,0.169607843137255,0.152941176470588,0.097058823529412,0.315686274509804

"Ilha do Frade",0.007915567282322,0.015831134564644,0.131926121372032,0.079155672823219,0.118733509234828,0.12664907651715,0.25065963060686,0.269129287598945

"Ilha do Príncipe",0.035541904344011,0.197016235190873,0.272926722246599,0.132514260640632,0.027643703378675,0.000438788942519,0,0.333918385256691

"Inhanguetá",0.030685402925151,0.167765987955262,0.286205907657012,0.111843991970175,0.018353885861772,0.002867794665902,0,0.382277028964726

"Itararé",0.021320093457944,0.196261682242991,0.263726635514019,0.145297897196262,0.032272196261682,0.003796728971963,0.000292056074766,0.337032710280374

"Jabour",0.004040404040404,0.067676767676768,0.17979797979798,0.223232323232323,0.171717171717172,0.056565656565657,0.002020202020202,0.294949494949495

"Jardim Camburí",0.005107572904183,0.057267591165896,0.115990412600582,0.249357986646122,0.208012326656394,0.083204930662558,0.018004907835416,0.263054271528848

"Jardim da Penha",0.005796691644281,0.063622225364061,0.108405202884208,0.222501060370423,0.202071256892408,0.10239643715538,0.027357556906546,0.267849568782695

"Jesus de Nazareth",0.027264325323475,0.237060998151571,0.230129390018484,0.092883548983364,0.015711645101664,0.002310536044362,0.000462107208872,0.394177449168207

"Joana D'arc",0.007311129163282,0.132818846466288,0.227457351746547,0.186027619821284,0.062144597887896,0.014216084484159,0.000406173842405,0.36961819658814

"Jucutuquara",0.013081395348837,0.12281976744186,0.199127906976744,0.243459302325581,0.131540697674419,0.031976744186047,0.009447674418605,0.248546511627907

"Maria Ortiz",0.023088403125274,0.182512509876218,0.271530155385831,0.170836625406022,0.035203230620665,0.007725397243438,0.001316829075586,0.307786849266965

"Mário Cypreste",0.005853658536585,0.205853658536585,0.336585365853659,0.113170731707317,0.014634146341463,0.001951219512195,0,0.321951219512195

"Maruípe",0.00998003992016,0.110921015112632,0.184203022526376,0.25007128599943,0.132021670943827,0.03222127174223,0.007128599942971,0.273453093812375

"Mata da Praia",0.002272257797976,0.03470357364181,0.064346209460855,0.140363561247676,0.187771121669077,0.162053294773807,0.096364387523239,0.312125593885561

"Monte Belo",0.011864406779661,0.189265536723164,0.236158192090395,0.187005649717514,0.066666666666667,0.013559322033898,0,0.295480225988701

"Morada de Camburí",0.005550416281221,0.074930619796485,0.137835337650324,0.195189639222942,0.175763182238668,0.094357076780759,0.039777983348751,0.276595744680851

"Nazareth",0.005607476635514,0.106542056074766,0.203738317757009,0.220560747663551,0.125233644859813,0.050467289719626,0.007476635514019,0.280373831775701

"Nova Palestina",0.036336109008327,0.254731264193793,0.23996971990916,0.077971233913702,0.010598031794095,0.002081756245269,0.000189250567752,0.378122634367903

"Parque Industrial",0,0.4,0.1,0.1,0,0,0,0.4

"Parque Moscoso",0.009840098400984,0.093480934809348,0.145141451414514,0.274292742927429,0.149446494464945,0.056580565805658,0.009840098400984,0.261377613776138

"Piedade",0.0187265917603,0.217228464419476,0.239700374531835,0.104868913857678,0.02247191011236,0.0187265917603,0.00749063670412,0.370786516853933

"Pontal de Camburí",0.008782936010038,0.092848180677541,0.150564617314931,0.234629861982434,0.117942283563363,0.040150564617315,0.012547051442911,0.342534504391468

"Praia do Canto",0.003726530027232,0.046366633223449,0.078042138454923,0.150709473985954,0.192776264870288,0.165973914289809,0.101691271320052,0.260713773828293

"Praia do Suá",0.027027027027027,0.186009538950715,0.172098569157393,0.140302066772655,0.079093799682035,0.04093799682035,0.013116057233704,0.341414944356121

"Redenção",0.023506743737958,0.254720616570328,0.260500963391137,0.098265895953757,0.010019267822736,0.000385356454721,0,0.352601156069364

"República",0.00721917412648,0.103378573491193,0.18336702281259,0.249783424776206,0.129945134276639,0.045625180479353,0.007796708056598,0.272884781980941

"Resistência",0.025761973875181,0.263969521044993,0.254716981132075,0.080370101596517,0.008708272859216,0.001269956458636,0.000362844702467,0.364840348330914

"Romão",0.036604361370717,0.252725856697819,0.229361370716511,0.103582554517134,0.017523364485981,0.004672897196262,0.001168224299065,0.354361370716511

"Santa Cecília",0.012121212121212,0.084848484848485,0.141125541125541,0.258874458874459,0.159307359307359,0.055411255411255,0.008658008658009,0.27965367965368

"Santa Clara",0.014904187366927,0.1291696238467,0.190205819730305,0.21930447125621,0.124201561391058,0.028388928317956,0.01277501774308,0.281050390347764

"Santa Helena",0.002333177788147,0.044330377974802,0.067662155856276,0.141390573961736,0.187587494167056,0.185254316378908,0.11012599160056,0.261315912272515

"Santa Lúcia",0.005927389478884,0.061990614966658,0.103976290442084,0.221289207211657,0.186712768584836,0.111138552729069,0.034576438626821,0.27438873795999

"Santa Luíza",0.001779359430605,0.086298932384342,0.133451957295374,0.190391459074733,0.158362989323843,0.094306049822064,0.034697508896797,0.300711743772242

"Santa Martha",0.026100236012773,0.198250728862974,0.252394835485214,0.160349854227405,0.038733860891295,0.005275579619603,0.001110648340969,0.317784256559767

"Santa Tereza",0.012486992715921,0.086715227193895,0.237599722511273,0.162330905306972,0.048213666319806,0.010405827263267,0.000346860908776,0.44190079778009

"Santo André",0.043271139341008,0.247717348154029,0.269948392219135,0.102024612941644,0.012703453751489,0.000793965859468,0,0.323541087733227

"Santo Antônio",0.014925373134328,0.173555300420972,0.238040566398775,0.179678530424799,0.044010715652507,0.012437810945274,0.002104860313816,0.335246842709529

"Santos Dumont",0.024451410658307,0.208777429467085,0.285266457680251,0.13166144200627,0.030094043887147,0.006269592476489,0,0.313479623824451

"Santos Reis",0.052023121387283,0.278612716763006,0.228901734104046,0.060115606936416,0.009248554913295,0.001156069364162,0,0.369942196531792

"São Benedito",0.036954585930543,0.227960819234194,0.214603739982191,0.076580587711487,0.009349955476402,0.000890471950134,0.000890471950134,0.432769367764915

"São Cristovão",0.013780059443394,0.18454471764388,0.225614698730073,0.190759254255607,0.062415563361254,0.009727100783572,0.002161577951905,0.310997027830316

"São José",0.027670171555064,0.237410071942446,0.271167681239624,0.092418372993913,0.012451577199779,0.000553403431101,0.000276701715551,0.358052019922523

"São Pedro",0.027131782945736,0.240863787375415,0.252214839424142,0.113233665559247,0.017995570321152,0.002768549280177,0,0.345791805094131

"Segurança do Lar",0.01980198019802,0.130693069306931,0.22970297029703,0.247524752475248,0.075247524752475,0.02970297029703,0.003960396039604,0.263366336633663

"Solon Borges",0.012536873156342,0.13716814159292,0.247050147492625,0.249262536873156,0.07669616519174,0.019174041297935,0.000737463126844,0.257374631268437

"Tabuazeiro",0.01665780613149,0.186425660109871,0.248803827751196,0.168172957646642,0.051213893319157,0.01010101010101,0.001594896331738,0.317029948608896

"Universitário",0.011568123393316,0.159383033419023,0.236503856041131,0.185089974293059,0.079691516709512,0.015424164524422,0.005141388174807,0.30719794344473

"Vila Rubim",0.022745098039216,0.188235294117647,0.224313725490196,0.179607843137255,0.063529411764706,0.012549019607843,0.00078431372549,0.308235294117647

""")



# This data contains how the incomes are distributed in

# each neighbourhood for people older than 10 yo.

# Incoming are splitted in 8 ranges.

# people without income are allocated in range_0

# people with incoming lower than half minimun wage are range_1

# and so on...

# The values are the percentages population in the respective range



income_ranges = pd.read_csv(ranges_csv, index_col=None)

income_ranges.rename(columns={"Mesorregiões, microrregiões, municípios, distritos, subdistritos e bairros":

                                'neighbourhood',

                            "Sem rendimento (2)": 'neigh_income_range_0',

                            "Até ½ salário mínimo": 'neigh_income_range_1',

                            "Mais de 1/2 a 1 salário mínimo": 'neigh_income_range_2',

                            "Mais de 1 a 2 salário mínimo": 'neigh_income_range_3',

                            "Mais de 2 a 5 salário mínimo": 'neigh_income_range_4',

                            "Mais de 5 a 10 salário mínimo": 'neigh_income_range_5',

                            "Mais de 10 a 20 salário mínimo": 'neigh_income_range_6',

                            "Mais de 20 salário mínimo": 'neigh_income_range_7',

                            },

                    inplace=True)
income_ranges.head()
def strip_accents(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



df.neighbourhood = df.neighbourhood.str.lower()

mean_income.neighbourhood = mean_income.neighbourhood.str.lower()

income_ranges.neighbourhood = income_ranges.neighbourhood.str.lower()



df.neighbourhood = df.neighbourhood.apply(strip_accents)

mean_income.neighbourhood = mean_income.neighbourhood.apply(strip_accents)

income_ranges.neighbourhood = income_ranges.neighbourhood.apply(strip_accents)
# aeroporto isn't a real neighbourhood, changing to nearest one

df.loc[df.neighbourhood == "aeroporto", "neighbourhood"] = "jardim camburi"

# Fixes apostrophe

df.loc[df.neighbourhood == "joana d´arc", "neighbourhood"] = "joana d'arc"

# removing 2 patients from island

df = df[df.neighbourhood != "ilhas oceanicas de trindade"]
df = pd.merge(df, mean_income, left_on="neighbourhood", right_on="neighbourhood", how='left', sort=False)

# Check if there is any missing neighbourhood

df.loc[df.neigh_mean_income.isnull(), ["neighbourhood", "neigh_mean_income"]]
df = pd.merge(df, income_ranges, left_on="neighbourhood", right_on="neighbourhood", how='left', sort=False)

# Check if there is any missing neighbourhood

df.loc[df.neigh_income_range_0.isnull(), ["neighbourhood", "neigh_income_range_0"]]
df.head()
df["no-show"] = df["no-show"].map({"Yes": True, "No": False})

df["show"] = ~df["no-show"]

del[df["no-show"]]
for feature in ["diabetes", "alcoholism", "hypertension", "handicap",

                "scholarship", "sms_received", "neighbourhood"]:

    print("{}: {}".format(feature, df[feature].unique()))
boolean_features = ["diabetes", "alcoholism", "hypertension", "sms_received", "scholarship"]

categorical_features = ["gender", "handicap", "neighbourhood", "patient_id", "appointment_id"]



df.age = df.age.astype("int")

df.patient_id = df.patient_id.astype("int")

df.appointment_id = df.appointment_id.astype("int")



for feature in boolean_features:

    df[feature] = df[feature].astype("bool")



for feature in categorical_features:

    df[feature] = df[feature].astype("category")
df.info()
df.head()
df.describe(include="all")
df["days_delta"] = (df.appointment_day - pd.to_datetime(df.scheduled_day.dt.date)).dt.days
# Using only prior appointments

def calculate_prior_noshow(row):

    previous_appoint = df.loc[(df.patient_id == row["patient_id"]) & (df.appointment_day <= row["scheduled_day"]), "show"]

    row["previous_appoint_count"] = len(previous_appoint)

    row["previous_appoint_shows"] = previous_appoint.sum()

    return row

          

df = df.apply(calculate_prior_noshow, axis=1)

df = df.drop(["patient_id", "appointment_id"], axis=1)
boolean_features = ["diabetes", "alcoholism", "hypertension", "sms_received", "scholarship"]

categorical_features = ["gender", "handicap", "neighbourhood"]



df.age = df.age.astype("int")



for feature in boolean_features:

    df[feature] = df[feature].astype("bool")



for feature in categorical_features:

    df[feature] = df[feature].astype("category")
df.loc[df.age < 0, "age"] = int(df.age.mode())

df.loc[df.days_delta < 0, "days_delta"] = int(df.days_delta.mode())
df["ln_days_delta"] = np.log(df.days_delta + 1)

df = df.drop(["days_delta"], axis=1)
plt.figure(figsize=(9.5,18))

gs = gridspec.GridSpec(4, 2)



features = df.select_dtypes(include=["category", "bool"]).columns.drop(["show"])



for i, feature in enumerate(features):

    ax = plt.subplot(gs[(i // 2), (i % 2)])

    sns.countplot(df[feature], ax=ax)



plt.tight_layout()

plt.show()
plt.figure(figsize=(9, 4.5))

sns.distplot(df.age, bins=df.age.max(), kde=False)

plt.tight_layout()

plt.show()
plt.figure(figsize=(9, 4.5))

sns.distplot(df.ln_days_delta, kde=False)

plt.tight_layout()

plt.show()
plt.figure(figsize=(9, 4.5))

sns.distplot(df.neigh_mean_income, kde=False)

plt.tight_layout()

plt.show()
plt.figure(figsize=(9.5,18))

gs = gridspec.GridSpec(4, 2)



features = df.select_dtypes(include=["category", "bool"]).columns.drop(["show"])



for i, feature in enumerate(features):

    

    feature_counts = (df.groupby([feature])["show"]

                            .value_counts(normalize=True)

                            .rename('frequence')

                            .reset_index()

                            .sort_values(feature))

    

    ax = plt.subplot(gs[(i // 2), (i % 2)])

    plot = sns.barplot(x=feature, y="frequence", hue="show",

                       data=feature_counts, ax=ax)

    plot.set_ylabel("Frequence")

    plot.set_xlabel(feature)



plt.tight_layout()

plt.show()
ax = sns.FacetGrid(df, hue="show", size=4.5, aspect=2)

ax = ax.map(sns.distplot, "age", bins=df.age.max(), kde=True)

ax.set_ylabels("Frequence")

plt.legend(loc='upper right')

plt.tight_layout()

plt.show()
ax = sns.FacetGrid(df, hue="show", size=4.5, aspect=2)

ax = ax.map(sns.distplot, "ln_days_delta", kde=True)

ax.set_ylabels("Frequence")

plt.legend(loc='upper left')

plt.tight_layout()

plt.show()
ax = sns.FacetGrid(df, hue="show", size=4.5, aspect=2)

ax = ax.map(sns.distplot, "neigh_mean_income", kde=True)

ax.set_ylabels("Frequence")

plt.legend(loc='upper left')

plt.tight_layout()

plt.show()
# transformimg date values into sin/cos



# hour of registration

angle = df.scheduled_day.dt.hour * (2 * np.pi) / 24

df["registration_hour_sin"] = np.sin(angle)

df["registration_hour_cos"] = np.cos(angle)



# Day of the week (appointment)

angle = df.appointment_day.dt.dayofweek * (2 * np.pi) / 7

df["appointment_week_sin"] = np.sin(angle)

df["appointment_week_cos"] = np.cos(angle)



# Day of the month (appointment)

angle = df.appointment_day.dt.day * (2 * np.pi) / df.appointment_day.dt.days_in_month

df["appointment_month_sin"] = np.sin(angle)

df["appointment_month_cos"] = np.cos(angle)



# Day of the year (appointment)

angle = df.appointment_day.dt.dayofyear * (2 * np.pi) / 365 

df["appointment_year_sin"] = np.sin(angle)

df["appointment_year_cos"] = np.cos(angle)



# Dropping registration date, appointment date and day_of_the_week

df = df.drop(["appointment_day", "scheduled_day"], axis=1)
df.head()
one_hot_features = pd.get_dummies(df.drop(["show", "neighbourhood"], axis=1)).columns

X = pd.get_dummies(df.drop(["show", "neighbourhood"], axis=1)).values

y = df.show.values



X = X.astype("float64")

y = y.astype("float64")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, random_state=7)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.5, random_state=7)
train_df = pd.DataFrame(data=X_train, columns=one_hot_features)

train_df["show"] = y_train

train_df.head()
corr_tresh = 2 / np.sqrt(X_train.shape[0])

print("Correlation treshold: {:.4f}".format(corr_tresh))
r = pd.DataFrame(train_df.corr())

output_r = pd.DataFrame(r.loc[:, "show"].abs())

output_r["correlated"] = output_r > corr_tresh

output_r
plt.figure(figsize=(9, 6))

plt.title("Pearson correlation between feature and output")

output_r.loc[output_r.index != "show", "show"].plot(kind="bar", label="")

tresh_line = plt.axhline(corr_tresh, linewidth=2, linestyle="--", color='r', label="2/√p")

plt.yscale("log")

plt.legend()

plt.tight_layout()

plt.show()
one_hot_features = pd.get_dummies(df.drop(["show", "neighbourhood", "alcoholism", "gender"], axis=1)).columns

X = pd.get_dummies(df.drop(["show", "neighbourhood", "alcoholism", "gender"], axis=1)).values

y = df.show.values



X = X.astype("float64")

y = y.astype("float64")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, random_state=7)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.5, random_state=7)
one_hot_features
scaled_features = np.flatnonzero(one_hot_features.str.startswith("neigh_income_range_") |

                                 one_hot_features.str.startswith("previous_appoint_") |

                                 one_hot_features.isin(["age", "ln_days_delta", "neigh_mean_income"]))



scaler = StandardScaler()

scaler.fit(X_train[:, scaled_features])
X_train[:, scaled_features] = scaler.transform(X_train[:, scaled_features])

X_val[:, scaled_features] = scaler.transform(X_val[:, scaled_features])

X_test[:, scaled_features] = scaler.transform(X_test[:, scaled_features])
X_train.shape, X_val.shape, X_test.shape
clf = LogisticRegression(class_weight="balanced")



clf_grid = GridSearchCV(estimator=clf,

                        param_grid={

                            "C": np.logspace(-3, 2, num=20),

                        },

                        cv=KFold(n_splits=5, shuffle=True),

                        scoring='roc_auc',

                        n_jobs=-1,

                        refit=True

)
clf_grid.fit(X_train, y_train)
plt.figure(figsize=(11,6))

plt.fill_between(

    list(clf_grid.cv_results_['param_C']),

    [(mean - std) for mean, std in zip(clf_grid.cv_results_["mean_test_score"], clf_grid.cv_results_["std_test_score"])],

    [(mean + std) for mean, std in zip(clf_grid.cv_results_["mean_test_score"], clf_grid.cv_results_["std_test_score"])],

    alpha=0.2

)

plt.plot(

    list(clf_grid.cv_results_['param_C']),

    [x for x in clf_grid.cv_results_["mean_test_score"]],

    '-o'

)

plt.axvline(clf_grid.best_params_["C"], color='r', alpha=0.5)

plt.xscale("log")

plt.xlabel("C")

plt.ylabel("Average AUC")

plt.title("Cross-validation search for C")

plt.show()
clf = clf_grid.best_estimator_
plt.figure(figsize=(6,6))

fpr, tpr, thresh = roc_curve(y_val, clf.predict_proba(X_val)[:,1])

roc_auc = auc(fpr, tpr)



plt.title('ROC Curve')

plt.plot(fpr, tpr, 'b', label="AUC: {:.3f}".format(roc_auc))

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.tight_layout()

plt.show()
y_prob = clf.predict_proba(X_val)[:, 1].squeeze()

prob_dist = pd.DataFrame({"prob": y_prob, "label": y_val.astype("bool")})

ax = sns.FacetGrid(prob_dist, hue="label", size=4.5, aspect=2)



ax = ax.map(sns.distplot, "prob", bins=100, kde=False)

plt.legend(title="show")

plt.xlabel("Classifier Probability")

plt.ylabel("Patients")

plt.tight_layout()

plt.show()
clf = XGBClassifier(n_estimators=400,

                    scale_pos_weight=((y_train == 0).sum() / y_train.sum()))

clf.silent = False
clf.fit(X_train, y_train)
clf.score(X_val, y_val)
plt.figure(figsize=(6,6))

fpr, tpr, thresh = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

roc_auc = auc(fpr, tpr)



plt.title('ROC Curve')

plt.plot(fpr, tpr, 'b', label="AUC: {:.3f}".format(roc_auc))

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.tight_layout()

plt.show()
features_importance = pd.Series(index=df.columns)



for feat in df.columns:

    feat_importance = one_hot_features.str.startswith(feat) * clf.feature_importances_

    features_importance[feat] = feat_importance.sum()



features_importance.sort_values(ascending=False, inplace=True)





plt.figure(figsize=(9, 6))

features_importance.plot(kind="bar")

plt.title("Feature Relevance")

plt.ylabel("Weight")

plt.grid(axis="y", color="k", linestyle="--")

plt.tight_layout()

plt.show()