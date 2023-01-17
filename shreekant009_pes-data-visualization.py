import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')



pes = pd.read_csv("../input/efootball-pes-2020-all-players-csv/deets-updated.csv")

pd.set_option('display.max_rows',20000, 'display.max_columns',20000)

pes.head()
Free_Agents=pes[pes['team_name'] == 'Free Agents']

Free_Agents_Names=Free_Agents['shirt_number'].tolist()

pes['shirt_number'] = pes['shirt_number'].replace(['Á. KOVÁCSIK',

 'A. STRUNA',

 'D. ILIEV',

 'J. KODJIA',

 'K. MBABU',

 'L. ŠATKA',

 'V. MAKSIMENKO',

 'C. MCLAUGHLIN',

 'R. KEOGH',

 'J. RUSSELL',

 'C. PHILIPPS',

 'A. ULMER',

 'K. BOUTAÏB',

 'K. ÁRNASON',

 'J. KANKAVA',

 'G. RÓLANTSSON',

 'W. GRIGG',

 'A. JUDGE',

 'N. ALIJI',

 'S. SLAVCHEV',

 'E. FORSBERG',

 'B. EMBOLO',

 'G. GHAZARYAN',

 'R. WILLIAMS',

 'B. MKHWANAZI',

 'H. KEKANA',

 'T. ZWANE',

 'S. RISVANIS',

 'A. ABRASHI',

 'M. DABBUR',

 'M. SABITZER',

 'P. SCHICK',

 'A. TINNERHOLM',

 'N. IOANNOU',

 'M. KORHUT',

 'O. DGANI',

 'Ž. TOMAŠEVIĆ',

 'A. RUSNÁK',

 'M. SIMIĆ',

 'B. BÜCHEL',

 'V. ANDRIUŠKEVIČIUS',

 'S. LARSSON',

 'M. BAGAYOKO',

 'M. SIPĽAK',

 'I. ŠEHIĆ',

 'F. MLADENOVIĆ',

 'G. SORČAN',

 'D. PALIAKOU',

 'M. MEDVEDEV',

 'G. GARAYEV',

 'A. POLO',

 'R. ALMEIDA',

 'D. BOHAR',

 'L. ZAHOVIČ',

 'N. SIHNEVICH',

 'D. PERETZ',

 'B. VALDEZ',

 'K. NUHU',

 'J. IKAUNIEKS',

 'A. KRAMARIĆ',

 'H. SHALA',

 'L. AUGUSTINSSON',

 'R. SHEIDAEV',

 'M. VELJKOVIC',

 'AMR AL SULAYA',

 'D. KEET',

 'T. HLATSHWAYO',

 'J. CALDERÓN',

 'E. DAVIS',

 'V. JOVOVIĆ',

 'S. POPOV',

 'T. NEDELEV',

 'Á. LANG',

 'Z. KALMÁR',

 'I. KOVÁCS',

 'D. DIBUSZ',

 'A. CALLENS',

 'T. TAWATHA',

 'M. KOZÁČIK',

 'Y. POULSEN',

 'J. PAVLENKA',

 'S. RUDY',

 'J. HECTOR',

 'A. ROMERO',

 'C. GONZÁLEZ',

 'C. PAVÓN',

 'D. SHISHKOVSKI',

 'E. LENJANI',

 'A. HOXHA',

 'H. HALLDÓRSSON',

 'Ö. KRISTINSSON',

 'E. HOVLAND',

 'R. BERIČ',

 'R. VASILEV',

 'J. GREGUŠ',

 'R. VARGA',

 'A. GABR',

 'K. MALINOV',

 'T. DOYLE',

 'C. HOWIESON',

 'M. CROCOMBE',

 'A. VUTOV',

 'R. CASTILLO',

 'J. VARGAS',

 'J. ORTIZ',

 'B. ACOSTA',

 'H. FIGUEROA',

 'A. MITCHELL',

 'C. LEWIS',

 'C. MANEA',

 'E. AGUILAR',

 'J. ONGUÉNÉ',

 'C. IKONOMIDIS',

 'S. BROTHERTON',

 'M. DYER',

 'M. MADISHA',

 'H. VACA',

 'M. RASHICA',

 'J. FUCHS',

 'G. LORIA',

 'M. AKANJI',

 'N. ELVEDI',

 'J. KNUDSEN',

 'ANDRÉ SILVA',

 'A. MITRIŢĂ',

 'B. GUSEYNOV',

 'G. ÁLVAREZ',

 'A. ARIAS',

 'É. FLORES',

 'J. ESPÍNOLA',

 'N. SCHULZ',

 'N. KOSOVIĆ',

 'M. EMRELI',

 'A. SANTAMARÍA',

 'S. BENEDETTINI',

 'T. KOUBEK',

 'D. SOW',

 'J. BREKALO',

 'A. FRANSSON',

 'R. STEFFEN',

 'I. DÍAZ',

 'G. ZHUKOV',

 'D. RAKELS',

 'A. ÖZBILIZ',

 'L. LAPPALAINEN',

 'N. KORZUN',

 'S. MAGOMEDALIYEV',

 'D. PETRAVIČIUS',

 'J. INGHAM',

 'M. JURMAN',

 'R. GRANT',

 'A. IVAN',

 'Y. MALLI',

 'D. NDLOVU',

 'A. TRAUSTASON',

 'A. ARROYO',

 'N. TZANEV',

 'H. CAMERON',

 'T. HUDSON-WIHONGI',

 'A. RUFER',

 'L. KLEINHEISLER',

 'E. ALVARADO',

 'A. ELIS',

 'B. BITTON',

 'J. RISDON',

 'R. OFORI',

 'MOHAMED HANY',

 'B. MAKENDZHIEV',

 'V. BOZHIKOV',

 'A. NEDYALKOV',

 'S. DELEV',

 'R. MATARRITA',

 'S. GALINDO',

 'J. MCLAUGHLIN',

 'J. DEMETRIOU',

 'J. BRUUN LARSEN',

 'D. ZAKARIA',

 'B. WOODBURN',

 'A. CIGAŅIKS',

 'V. JAGODINSKIS',

 'E. BENEDETTINI',

 'E. ŠETKUS',

 'A. HAKIMI',

 'S. MEMIŠEVIĆ',

 'F. JENSEN',

 'V. SELIMOVIC',

 'F. BERARDI',

 'J. MACLAREN',

 'D. INGHAM',

 'F. RØNNOW',

 'F. ESCOBAR',

 'K. PIRIĆ',

 'A. WARDA',

 'MAHMOUD GENNESH',

 'M. DEGENEK',

 'P. VINÍCIUS',

 'F. CRISANTO',

 'M. CHIRINOS',

 'E. GREZDA',

 'S. LAINER',

 'F. GRILLITSCH',

 'M. GAĆINOVIĆ',

 'R. CORDANO',

 'L. HAQUIN',

 'M. CUELLAR',

 'R. VACA',

 'L. VARGAS',

 'R. MPHAHLELE',

 'S. MABUNDA',

 'L. KLOSTERMANN',

 'W. ORBAN',

 'T. WERNER',

 'Z. MUSCAT',

 'JOSÉ RODRÍGUEZ',

 'D. MAN',

 'Y. MVOGO',

 'A. PRIJOVIĆ',

 'J. MIHALÍK',

 'K. LAIMER',

 'R. KIRILOV',

 'A. GEORGIOU',

 'J. SANCHO',

 'K. METS',

 'B. JOHNSEN',

 'P. MALONG',

 'M. WÖBER',

 'M. LÓPEZ',

 'C. KYRIAKOU',

 'M. KÄIT',

 'L. JANS',

 'N. AMPOMAH',

 'R. LEAL',

 'B. HALIMI',

 'P. ŠAFRANKO',

 'J. EDMUNDSSON',

 'M. GEORGE',

 'L. GERSON',

 'R. TAYLOR',

 'A. ŠĆEKIĆ',

 'S. PALITSEVICH',

 'C. HAZARD',

 'T. OKRIASHVILI',

 'G. NAVALOVSKI',

 'W. COULIBALY',

 'F. KAINZ',

 'X. SCHLAGER',

 'S. HLANTI',

 'I. MAELA',

 'T. MOKOENA',

 'T. LORCH',

 'A. TAGNAOUTI',

 'A. EL KAABI',

 'J. GULLEY',

 'A. DE JONG',

 'M. RIDENTON',

 'M. BEVAN',

 'T. HAMED',

 'L. TAHA',

 'N. MITROVIČ',

 'I. SMITH',

 'J. CANDÍA',

 'N. BANCU',

 'A. CICÂLDĂU',

 'V. KJOSEVSKI',

 'E. SARIĆ',

 'G. ZAKARIĆ',

 'A. BOICIUC',

 'V. DAVIDSEN',

 'A. PUTSILA',

 'V. GVILIA',

 'Y. SEIDAKHMET',

 'N. VALSKIS',

 'K. PIĄTEK',

 'N. KATCHARAVA',

 'J. THOMSEN',

 'A. SAROKA',

 'J. LAWRENCE',

 'S. NATTESTAD',

 'N. NANNI',

 'J. PIRINEN',

 'A. GRANLUND',

 'N. KVEKVESKIRI',

 'O. KITEISHVILI',

 'R. KVASKHVADZE',

 'L. DVALI',

 'A. MOUELHI',

 'A. PRIESTLEY',

 'J. CHIPOLINA',

 'R. STYCHE',

 'A. BARDON',

 'M. CAFER',

 'D. COLEING',

 'A. PONS',

 'E. BARNETT',

 'J. SERGEANT',

 'A. HERNANDEZ',

 'J. COOMBES',

 'K. GOLDWIN',

 'L. WALKER',

 'JC. GARCIA',

 'E. BRITTO',

 'T. DE BARR',

 'L. ANNESLEY',

 'E. JOLLEY',

 'A. HERNANDEZ',

 'J. MASCARENHAS',

 'R. CHIPOLINA',

 'L. CASCIARO',

 'T. HALL',

 'S. BENSI',

 'M. CHANOT',

 'D. TURPEL',

 'T. KIPS',

 'D. SINANI',

 'R. SCHON',

 'D. CARLSON',

 'M. DEVILLE',

 'L. BARREIRO',

 'D. SHCHARBITSKI',

 'P. SAVITSKI',

 'S. DRAHUN',

 'I. MAEUSKI',

 'A. RIOS',

 'G. DAGHBASHYAN',

 'K. HOVHANNISYAN',

 'H. HAMBARDZUMYAN',

 'A. BEGLARYAN',

 'A. KHACHATUROV',

 'S. ADAMYAN',

 'A. HAYRAPETYAN',

 'A. GRIGORYAN',

 'A. YEDIGARYAN',

 'T. BARSEGHYAN',

 'L. KASTRATI',

 'F. ALITI',

 'V. BEKAJ',

 'L. PAQARADA',

 'I. KOUSOULOS',

 'A. MAKRIS',

 'R. MARGAÇA',

 'C. WHEELER',

 'C. PANAYI',

 'B. KOPITOVIĆ',

 'M. IVANIĆ',

 'D. LJULJANOVIĆ',

 'M. MIJATOVIĆ',

 'R. RADUNOVIĆ',

 'S. MUGOŠA',

 'F. STOJKOVIĆ',

 'K. VELKOSKI',

 'B. NIKOLOV',

 'K. RISTEVSKI',

 'F. HASANI',

 'K. TOSHEVSKI',

 'E. BEJTULAI',

 'S. SPIROVSKI',

 'V. MUSLIU',

 'R. MUSCAT',

 'P. FENECH',

 'S. BORG',

 'J. ZERAFA',

 'C. FAILLA',

 'K. NWOKO',

 'J. GRECH',

 'J. MBONG',

 'D. VELLA',

 'J. CORBALAN',

 'B. KRISTENSEN',

 'H. BONELLO',

 'R. CAMILLERI',

 'MARC FERRER',

 'SEBASTIÁ GÓMEZ',

 'V. RODRÍGUEZ',

 'FERRAN POL',

 'R. FERNANDEZ',

 'JORDI ALÁEZ',

 'A. MARTINEZ',

 'JOAN CERVÓS',

 'EMILI GARCIA',

 'C. MARTÍNEZ',

 'M. SAN NICOLÁS',

 'MARC GARCÍA',

 'MAX LLOVERA',

 'M. VIHMANN',

 'B. LEPISTU',

 'T. TENISTE',

 'M. ROOSNUPP',

 'J. TAMM',

 'G. KAMS',

 'F. LIIVAK',

 'A. DMITRIJEV',

 'E. COCIUC',

 'C. SANDU',

 'A. ROZGONIUC',

 'G. ANTON',

 'R. GÎNSARI',

 'I. JARDAN',

 'D. GRAUR',

 'V. POSMAC',

 'M. KUKLYS',

 'L. KLIMAVIČIUS',

 'M. VOROBJOVAS',

 'D. BARTKUS',

 'V. SLAVICKAS',

 'A. JANKAUSKAS',

 'D. KAZLAUSKAS',

 'E. ZUBAS',

 'E. VAITKŪNAS',

 'R. BARAVYKAS',

 'A. ŽULPA',

 'E. GIRDVAINIS',

 'D. IKAUNIEKS',

 'A. KARAŠAUSKS',

 'K. DUBRA',

 'V. ISAJEVS',

 'K. IKSTENS',

 'I. TARASOVS',

 'R. RUGINS',

 'P. ŠTEINBORS',

 'R. SAVAĻNIEKS',

 'V. ŠABALA',

 'V. GABOVS',

 'L. MEIER',

 'F. EBERLE',

 'D. KAUFMANN',

 'S. YILDIZ',

 'D. BRÄNDLE',

 'S. WOLFINGER',

 'A. SELE',

 'T. HOBI',

 'A. MAJER',

 'M. GÖPPEL',

 'J. HOFER',

 'A. MALIN',

 'L. TOSI',

 'E. GOLINUCCI',

 'L. LUNADEI',

 'A. HIRSCH',

 'M. BATTISTINI',

 'M. CEVOLI',

 'M. MULARONI',

 'A. GOLINUCCI',

 'M. ZAVOLI',

 'A. GRANDONI',

 'D. CESARINI',

 'F. TOMASSINI',

 'M. BATTISTINI',

 'M. PALAZZI',

 'P. PASHAYEV',

 'R. DADAŞOV',

 'T. KHALILZADE',

 'R. MAMMADOV',

 'M. ABBASOV',

 'S. AGHAYEV',

 'D. NAZAROV',

 'A. ABDULLAYEV',

 'E. BALAYEV',

 'Á. FREDERIKSBERG',

 'M. EGILSSON',

 'O. FÆRØ',

 'J. BJARTALÍÐ',

 'H. VATNSDAL',

 'M. OLSEN',

 'K. JOENSEN',

 'K. Í BARTALSSTOVU',

 'R. JOENSEN',

 'G. VATNHAMAR',

 'K. OLSEN',

 'T. GESTSSON',

 'R. BALDVINSSON',

 'S. VATNHAMAR',

 'S. MALIY',

 'B. TURYSBEK',

 'G. SUYUMBAYEV',

 'D. NEPOGODOV',

 'R. MURTAZAYEV',

 'B. ISLAMKHAN',

 'S. MUZHIKOV',

 'T. ERLANOV',

 'Y. LOGVINENKO',

 'O. OMIRTAYEV',

 'A. BEYSEBEKOV',

 'M. FEDIN',

 'Y. PERTSUKH',

 'D. SHOMKO',

 'E. ĆIVIĆ',

 'P. NIAKHAICHYK',

 'D. LAPTSEU',

 'T. MKHIZE',

 'S. XULU',

 'L. MABOE',

 'M. BLAŽIČ',

 'G. VALERIANOS',

 'H. FONSECA',

 'D. TORRES',

 'D. MALDONADO',

 'J. ALVAREZ',

 'D. FLORES',

 'K. LÓPEZ',

 'F. HASANI',

 'N. FERRARESI',

 'S. MUSTAFÁ',

 'R. FERNÁNDEZ',

 'A. JUSINO',

 'C. ARANO',

 'L. VACA',

 'N. MICHAEL',

 'M. ANTONIOU',

 'F. PAPOULIS',

 'V. BOROVSKIJ',

 'M. PALIONIS',

 'P. GOLUBICKAS',

 'A. KRYVOTSYUK',

 'S. RAHIMOV',

 'MARC REBÉS',

 'CHUS RUBIO',

 'AARON SANCHEZ',

 'LUDOVIC CLEMENTE',

 'M. AKSALU',

 'N. BARANOV',

 'K. KALLASTE',

 'A. PIKK',

 'R. SAPPINEN',

 'S. ANDERSSON',

 'O. LAIZĀNS',

 'K. TOBERS',

 'V. GUTKOVSKIS',

 'R. GUBSER',

 'S. KÜHNE',

 'P. OSPELT',

 'L. VIGOUROUX',

 'B. BARÁTH',

 'D. SZOBOSZLAI',

 'D. TODOROVIĆ',

 'E. KOLJIČ',

 'Y. GERAFI',

 'A. HABASHI',

 'V. ANTOV',

 'G. IVANOV',

 'M. MINCHEV',

 'S. KOSTOV',

 'J. DAWA',

 'D. VELKOVSKI',

 'K. MARKOSKI',

 'C. STANKOVIC',

 'R. STREBINGER',

 'S. POSCH',

 'S. ARZAMENDIA',

 'K. MALGET',

 'F. ANNAN',

 'D. CELEADNIC',

 'V. AMBROS',

 'S. PLĂTICĂ',

 'A. BAADI',

 'N. STARK',

 'MOHAMED ABOUGABAL',

 'AHMED FATTOH',

 'MAHMOUD ALAA',

 'BAHER EL MOHAMADY',

 'ISLAM GABER',

 'AMAR HAMDI',

 'SALAH MOHSEN',

 'MOSTAFA MOHAMED',

 'N. VASILJEVIĆ',

 'N. VELLA',

 'K. MICALLEF',

 'M. GUILLAUMIER',

 'J. MINTOFF',

 'L. MONTEBELLO',

 'C. BROLLI',

 "A. D'ADDARIO",

 'O. MOSQUERA',

 'C. BLACKMAN',

 'E. WALKER',

 'O. BROWNE',

 'J. FAJARDO',

 'P. BRÅTVEIT',

 'L. CACACE',

 'N. BILLINGSLEY',

 'G. MELIKSETYAN',

 'A. CALISIR',

 'H. ISHKHANYAN',

 'A. MANUCHARYAN',

 'A. ISMAJLI',

 'M. UZUNI',

 'A. CRUZ',

 'K. FULLER',

 'R. ARAYA',

 'A. CRUZ',

 'J. MARÍN',

 'J. MORA',

 'A. LASSITER',

 'S. POKATILOV',

 'I. SHATSKIY',

 'E. AKHMETOV',

 'Y. VOROGOVSKIY',

 'A. HUTAR',

 'D. ONTUŽĀNS',

 'K. LAUKŽEMIS',

 'RENAT DADAŞOV',

 'A. RAMAZANOV',

 'M. MARTINS',

 'E. HAALAND',

 'B. MUSTAFAZADE',

 'R. KOCH',

 'R. KRUSE',

 'C. KAMENI',

 'R. JARSTEIN',

 'M. FIGUEROA',

 'G. TORRES',

 'M. BENATIA',

 'M. MILLIGAN',

 'G. JARA',

 'C. MULGREW',

 'M. BOUSSOUFA',

 'J. TOIVIO',

 'A. GRANQVIST',

 'G. SVENSSON',

 'A. JAAKKOLA',

 'K. LAFFERTY',

 'J. BŁASZCZYKOWSKI',

 'D. SIMONCINI',

 'S. MIKOLIŪNAS',

 'J. HABER',

 'M. MIFSUD',

 'JOSEP GÓMES',

 'MARC PUJOL',

 'MARCIO VIEIRA',

 'SERGI MORENO',

 'MARTIN BÜCHEL',

 'M. SUCHÝ',

 'N. MIHAYLOV',

 'B. KAYAL',

 'Y. SOMMER',

 'M. JØRGENSEN',

 'G. NIELSEN',

 'N. AMRABAT',

 'K. NÉMETH',

 'L. MCHEDLIDZE',

 'B. DZSUDZSÁK',

 'A. GUNNARSSON',

 'I. TRICHKOVSKI',

 'I. STASEVICH',

 'D. DA MOTA',

 'ILDEFONS LIMA',

 'I. KHUNE',

 'T. DANGDA',

 'A. WITSEL',

 'P. ZANEV',

 'C. MOŢI',

 'A. RUKAVINA',

 'MARCELO MORENO',

 'B. RUIZ',

 'E. IZAGUIRRE',

 'G. MCAULEY',

 'T. GEBRE SELASSIE',

 'M. SUMUSALO',

 'T. ELYOUNOUSSI',

 'C. DEAC',

 'A. MORIS',

 'T. KÁDÁR',

 'P. GULÁCSI',

 'A. DJOUM',

 'I. ARMAŞ',

 'A. MEHMEDI',

 'C. PANTILIMON',

 'P. PEKARÍK',

 'B. SÆVARSSON',

 'K. VASSILJEV',

 'M. POLVERINO',

 'C. LAMPE',

 'I. SHITOV',

 'S. PURI',

 'S. ZENJOV',

 'G. MEREBASHVILI',

 'Á. SZALAI',

 'A. AGIUS',

 'J. CARUANA',

 'S. NAMAŞCO',

 'E. CEBOTARU',

 'M. VITAIOLI',

 'D. RINALDI',

 'MARC VALES',

 'F. BEĆIRAJ',

 'T. DELANEY',

 'H. NORDTVEIT',

 'J. ANANIDZE',

 'D. GRIGORE',

 'V. BELEC',

 'H. PÉREZ',

 'Y. KHACHERIDI',

 'C. ORTIZ',

 'R. GERSHON',

 'K. THAMSATCHANAN',

 'A. AVRAAM',

 'M. RECHSTEINER',

 'S. KISLYAK',

 'J. ZOUA',

 'P. ARAJUURI',

 'A. GREGERSEN',

 'K. MKRTCHYAN',

 'K. HÄMÄLÄINEN',

 'M. JÄNISCH',

 'R. TORRES',

 'B. NATCHO',

 'A. BAROJA',

 'A. CARRILLO',

 'I. PIRIS',

 'M. VEŠOVIĆ',

 'P. KADEŘÁBEK',

 'A. ANTONIUC',

 'C. MAVRIAS',

 'G. KASHIA',

 'S. WIESER',

 'L. MEJÍA',

 'E. MAKHMUDOV',

 'E. KAÇE',

 'A. MACHADO',

 'A. GODOY',

 'A. COOPER',

 'Y. MOVSISYAN',

 'N. BODUROV',

 'S. LEPMETS',

 'D. RUBIO',

 'N. CASTILLO',

 'P. GALLESE',

 'A. COHEN',

 'J. CAMPBELL',

 'M. WAKASO',

 'A. AJETI',

 'M. MEERITS',

 'V. QAZAISHVILI',

 'T. SERERO',

 'J. HUŠBAUER',

 'A. GIANNOU',

 'R. QUAISON',

 'R. LOD',

 'B. YAYA',

 'R. SCHÜLLER',

 'L. JUSTINIANO',

 'A. CHUMACERO',

 'R. RAMALLO',

 'I. LICHNOVSKY',

 'O. MURILLO',

 'A. AL SHENAWY',

 'M. REUS',

 'B. BUTKO',

 'M. BOXALL',

 'A. CHIPCIU',

 'V. DARIDA',

 'Y. YOTÚN',

 'G. GONZÁLEZ',

 'H. LINDNER',

 'H. OJAMAA','R. QUIOTO','E. CAN','F. KOSTIĆ','B. BALAJ','A. FINNBOGASON','V. PÁLSSON','ALI GHAZAL','F. CALVO','R. SIGURJÓNSSON','T. JEDVAJ',

 'J. MORENO','D. BEJARANO','G. WHELAN','P. FORSELL','M. PÁTKAI','S. YEINI','M. ĎURIŠ','ANDRAŽ STRUNA','C. SONGKRASIN',

 'O. GABER',

 'L. GARRIDO',

 'T. PAYNE',

 'Y. REYNA',

 'D. MICHA',

 'M. LECKIE',

 'M. LANGERAK',

 'H. AFFUL',

 'T. SAINSBURY',

 'A. JĘDRZEJCZYK',

 'M. HINTEREGGER',

 'S. ILSANKER',

 'J. MONTES',

 'DOSSA JÚNIOR',

 'G. VASILIOU',

 'N. MYTIDES',

 'P. SOTERIOU',

 'K. ARTYMATAS',

 'F. MORA',

 'D. VALDÉS',

 'Á. SAGAL',

 'R. BÜRKI',

 'M. LANG',

 'S. ZUBER',

 'K. AYHAN',

 'G. LOVRENCSICS',

 'D. KOWNACKI',

 'R. GUERREIRO',

 'G. EFREM',

 'J. MARTÍNEZ',

 'G. MERKIS',

 'V. STOYANOV',

 'T. BUNMATHAN',

 'L. MOREIRA',

 'B. BECKELES',

 'B. SANGARÉ',

 'H. VILLALBA',

 'R. SARAVIA',

 'M. MEZA',

 'O. BEN HARUSH',

 'H. ANIER',

 'S. VEGAS',

 'J. BRANDT',

 'W. KANON',

 'S. GBOHOUO',

 'A. CISSÉ',

 'E. BIČAKČIĆ',

 'R. CASTRO',

 'E. CARDONA',

 'C. CÁCEDA',

 'E. SAAVEDRA',

 'L. OVALLE',

 'M. GINTER',

 'K. WASTON',

 'L. LOPEZ',

 'ALEXANDER LÓPEZ',

 'B. ROCHEZ',

 'B. HENDRIKSSON',

 'T. VERMAELEN',

 'R. ROJAS',

 'U. PARDO',

 'A. HLEB',

 'K. CASTEELS',

 'M. BEJARANO',

 'T. HAZARD',

 'A. NABBOUT',

 'J. JEGGO',

 'A. FIGUERA',

 'R. GÍSLASON',

 'A. HUGHES',

 'C. KEŞERÜ',

 'K. EL AHMADI',

 'S. LARSSON']

,-1)

#Missing Value Treatment

pes['playing_style'] = pes['playing_style'].fillna(value='NO INFO')

pes['playing_style'] = pes['playing_style'].replace({'---':'NO INFO'})
pes.isnull().sum()
fig = px.treemap(pes, path=['league','team_name','name','overall_rating'],color='league')

fig.update_layout(

    title='Detail Player Information')

fig.show()
pes_sum_or = pes.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['overall_rating']=pes_count_or['Total Players']

Worlds_Best_Team=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Worlds_Best_Team['team_name'][:50], y=Worlds_Best_Team['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,0,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=7))

fig.update_layout(title_text='Top 50 Worlds Best Football Team')

fig.show()
#Europe

pes_europe = pes[pes['region']=='Europe']

pes_player_detail= pes_europe.groupby(['region','nationality','registered_position','team_name'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position','team_name'],values='Player_Count')

fig.update_layout(

    title='European Player with Country,Playing Position and for the CLUB they play for')

fig.show()
#South America

pes_south_america = pes[pes['region']=='South America']

pes_player_detail= pes_south_america.groupby(['region','nationality','registered_position','team_name'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position','team_name'],values='Player_Count')

fig.update_layout(

    title='South American Player with Country,Playing Position and for the CLUB they play for')

fig.show()
#Africa

pes_africa = pes[pes['region']=='North & Central America']

pes_player_detail= pes_africa.groupby(['region','nationality','registered_position','team_name'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position','team_name'],values='Player_Count')

fig.update_layout(

    title='North & Central America Player with Country,Playing Position and for the CLUB they play for')

fig.show()
#Africa

pes_africa = pes[pes['region']=='Africa']

pes_player_detail= pes_africa.groupby(['region','nationality','registered_position','team_name'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position','team_name'],values='Player_Count')

fig.update_layout(

    title='African Player with Country,Playing Position and for the CLUB they play for')

fig.show()
#ASIA

pes_asia = pes[pes['region']=='Asia-Oceania']

pes_player_detail= pes_asia.groupby(['region','nationality','registered_position','team_name'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position','team_name'],values='Player_Count')

fig.update_layout(

    title='Asia-Oceanic Player with Country,Playing Position and for the CLUB they play for')

fig.show()
fig = px.pie(pes, names='ball_color',hole=.3)

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=0.9)))

fig.update_layout(

    title='Player Count Divided in from White(1-Star Player) to Black(5-Star Player)')

fig.show()
fig = px.histogram(pes,x='region')

fig.update_traces(marker_color='rgb(80,10,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Continentwise Players in PES2020')

fig.show()
pes_sum_or = pes.groupby('league')['overall_rating'].sum().reset_index()

pes_count_or = pes.groupby('league')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Z=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['league','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Z['league'], y=Z['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(250,10,50)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Best League in the world based on Overall Player Rating')

fig.show()
fig = px.histogram(pes,x='league')

fig.update_traces(marker_color='rgb(255,0,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Number of Players in Every League')

fig.show()
pes['age'] = pd.to_numeric(pes['age'], errors='coerce')

fig = px.histogram(pes,x='age')

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(size=14))

fig.update_layout(

    title='Agewise Count of Players')

fig.show()
fig = px.box(pes, x=pes['region'], y=pes['age'])

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Age Distribution of Players with Continent in PES2020')

fig.show()
fig = px.box(pes, x=pes['nationality'], y=pes['age'])

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.update_layout(

    title='Age Distribution of Players with Nationality in PES2020')

fig.show()
fig = px.box(pes, x=pes['league'], y=pes['age'])

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=12))

fig.update_layout(

    title='Age Distribution of Players with Club in PES2020')

fig.show()
pes['weight'] = pd.to_numeric(pes['weight'], errors='coerce')

fig = px.histogram(pes,x='weight')

fig.update_traces(marker_color='rgb(210,220,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(

    title='Weightwise Count of Players')

fig.show()
fig = px.box(pes, x=pes['region'], y=pes['weight'])

fig.update_traces(marker_color='rgb(210,220,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Weight Distribution of Players with Continent in PES2020')

fig.show()
fig = px.box(pes, x=pes['nationality'], y=pes['weight'])

fig.update_traces(marker_color='rgb(210,220,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.update_layout(

    title='Weight Distribution of Players with Nationality in PES2020')

fig.show()
fig = px.box(pes, x=pes['league'], y=pes['weight'])

fig.update_traces(marker_color='rgb(210,220,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=12))

fig.update_layout(

    title='Weight Distribution of Players with Continent in PES2020')

fig.show()
pes['height'] = pd.to_numeric(pes['height'], errors='coerce')

fig = px.histogram(pes,x='height')

fig.update_traces(marker_color='rgb(100,210,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(size=14))

fig.update_layout(

    title='Heightwise(cm) Count of Players')

fig.show()
fig = px.box(pes, x=pes['region'], y=pes['height'])

fig.update_traces(marker_color='rgb(100,210,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Height Distribution of Players with Continent in PES2020')

fig.show()
fig = px.box(pes, x=pes['nationality'], y=pes['height'])

fig.update_traces(marker_color='rgb(100,210,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.update_layout(

    title='Height Distribution of Players with Nationality in PES2020')

fig.show()
fig = px.box(pes, x=pes['league'], y=pes['height'])

fig.update_traces(marker_color='rgb(100,210,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=12))

fig.update_layout(

    title='Height Distribution of Players with Continent in PES2020')

fig.show()
fig = px.histogram(pes,x='foot')

fig.update_traces(marker_color='rgb(10,255,10)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(size=14))

fig.update_layout(

    title='Foot Prefferece by the Players in PES2020')

fig.show()
fig = px.pie(pes,names='playing_style', title='Player Playing Style in PES2020')

fig.update_traces(hole=.3, hoverinfo="label+percent+name")

fig.show()
fig = px.histogram(pes,x='registered_position')

fig.update_traces(marker_color='rgb(10,10,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))



fig.update_layout(

    title='Registered Position by the Players in PES2020')

fig.show()
print("                                    ")

print('______________________________________________________________________________')

print("Youngest Players in PES 2020:-")

print("                                    ")

pes_youngest=pes[pes['age']== 15.0]

y=pes_youngest.sort_values('age',ascending=True)[['name','nationality','league','team_name','registered_position','age']]

print(y)

print('______________________________________________________________________________')

print("                                    ")

print('______________________________________________________________________________')

print("Oldest Players in PES 2020:-")

print("                                    ")

pes_oldest=pes[pes['age']== 43.0]

y=pes_oldest.sort_values('age',ascending=False)[['name','nationality','league','team_name','registered_position','age']]

print(y)

print('______________________________________________________________________________')

print("                                    ")
print('_________________________________________________________________________________')

print("Shortest Player in PES 2020:-")

print("                                    ")

pes_shortest=pes[pes['height']== 148.0]

y=pes_shortest.sort_values('height',ascending=True)[['name','nationality','league','region','team_name','registered_position','height']]

print(y)

print('_________________________________________________________________________________')

print("                                    ")

print('_________________________________________________________________________________')

print("Tallest Player in PES 2020:-")

print("                                    ")

pes_tallest=pes[pes['height']== 207.0]

y=pes_tallest.sort_values('height',ascending=False)[['name','nationality','league','region','team_name','registered_position','height']]

print(y)

print('_________________________________________________________________________________')
print("                                    ")

print('_________________________________________________________________________________')

print("Thinnest Player in PES 2020:-")

print("                                    ")

pes_thinest=pes[pes['weight']== 45.0]

y=pes_thinest.sort_values('weight',ascending=True)[['name','nationality','league','region','team_name','registered_position','weight']]

print(y)

print('_________________________________________________________________________________')

print("                                    ")

print('_________________________________________________________________________________')

print("Heaviest Player in PES 2020:-")

print("                                    ")

pes_heaviest=pes[pes['weight']== 113.0]

y=pes_heaviest.sort_values('weight',ascending=False)[['name','nationality','league','region','team_name','registered_position','weight']]

print(y)

print('_________________________________________________________________________________')
fig = px.box(pes, y=pes['overall_rating'], x=pes['age'])

fig.update_layout(

    title='Agewise Growth and Decline of Player')

fig.update_traces(marker_color='rgb(110,255,110)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.show()
x0 = pes['weak_foot_usage']

x1 = pes['weak_foot_accuracy']

group_labels = ['weak_foot_usage', 'weak_foot_accuracy']

fig = go.Figure()

fig.add_trace(go.Histogram(x=x0),)

fig.add_trace(go.Histogram(x=x1))



fig.update_layout(title_text='Players with Weak Foot:Blue = weak_foot_usage & Red = weak_foot_accuracy')

fig.update_layout(barmode='stack')

fig.show()
pes_best_players = pes.iloc[pes.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

pes_best_players
pes_worst_players = pes.iloc[pes.groupby(['registered_position'])['overall_rating'].idxmin()][['name','nationality','team_name','registered_position','overall_rating']]

pes_worst_players
English_League = pes[pes['league']=='English League']

English_League_XI = English_League.loc[English_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

English_League_XI
pes_english_league = pes[pes['league']=='English League']

pes_sum_or = pes_english_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_english_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

English_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=English_League['team_name'], y=English_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(0,102,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of English League Team')

fig.show()
pes_english_league = pes[pes['league']=='English League']

pes_sum_form = pes_english_league.groupby('team_name')['form'].sum().reset_index()

pes_count_form = pes_english_league.groupby('team_name')['form'].count().reset_index(name='Total Players')

pes_sum_form['Average overall_rating'] = pes_sum_form['form']/pes_count_form['Total Players']

pes_sum_form['form']=pes_count_or['Total Players']

English_League=pes_sum_form.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=English_League['team_name'], y=English_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(0,102,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Form of English League Team')

fig.show()
Spanish_League = pes[pes['league']=='Spanish League']

Spanish_League_XI = Spanish_League.loc[Spanish_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

Spanish_League_XI
pes_spanish_league = pes[pes['league']=='Spanish League']

pes_sum_or = pes_spanish_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_spanish_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Spanish_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Spanish_League['team_name'], y=Spanish_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,64,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of Spanish League Team')

fig.show()
pes_spanish_league = pes[pes['league']=='Spanish League']

pes_sum_form = pes_spanish_league.groupby('team_name')['form'].sum().reset_index()

pes_count_form = pes_spanish_league.groupby('team_name')['form'].count().reset_index(name='Total Players')

pes_sum_form['Average overall_rating'] = pes_sum_form['form']/pes_count_form['Total Players']

pes_sum_form['form']=pes_count_or['Total Players']

Spanish_League=pes_sum_form.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Spanish_League['team_name'], y=Spanish_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,64,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Form of Spanish League Team')

fig.show()
French_League = pes[pes['league']=='Ligue 1 Conforama']

French_League_XI = French_League.loc[French_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

French_League_XI
pes_french_league = pes[pes['league']=='Ligue 1 Conforama']

pes_sum_or = pes_french_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_french_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

French_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=French_League['team_name'], y=French_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(110,255,110)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of French League Team')

fig.show()
pes_french_league = pes[pes['league']=='Ligue 1 Conforama']

pes_sum_form = pes_french_league.groupby('team_name')['form'].sum().reset_index()

pes_count_form = pes_french_league.groupby('team_name')['form'].count().reset_index(name='Total Players')

pes_sum_form['Average overall_rating'] = pes_sum_form['form']/pes_count_form['Total Players']

pes_sum_form['form']=pes_count_or['Total Players']

French_League=pes_sum_form.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=French_League['team_name'], y=French_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(110,255,110)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Form of French League Team')

fig.show()
Italian_League = pes[pes['league']=='Serie A TIM']

Italian_League_XI = Italian_League.loc[Italian_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

Italian_League_XI
pes_italian_league = pes[pes['league']=='Serie A TIM']

pes_sum_or = pes_italian_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_italian_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Italian_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Italian_League['team_name'], y=Italian_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,0,191)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of Italian League Team')

fig.show()
pes_italian_league = pes[pes['league']=='Serie A TIM']

pes_sum_form = pes_italian_league.groupby('team_name')['form'].sum().reset_index()

pes_count_form = pes_italian_league.groupby('team_name')['form'].count().reset_index(name='Total Players')

pes_sum_form['Average overall_rating'] = pes_sum_form['form']/pes_count_form['Total Players']

pes_sum_form['form']=pes_count_or['Total Players']

Italian_League=pes_sum_form.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Italian_League['team_name'], y=Italian_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,0,191)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Form of Italian League Team')

fig.show()
Dutch_League = pes[pes['league']=='Eredivisie']

Dutch_League_XI = Dutch_League.loc[Dutch_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

Dutch_League_XI
pes_dutch_league = pes[pes['league']=='Eredivisie']

pes_sum_or = pes_dutch_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_dutch_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Dutch_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Dutch_League['team_name'], y=Dutch_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,255,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of Eredivisie League Team')

fig.show()
pes_dutch_league = pes[pes['league']=='Eredivisie']

pes_sum_form = pes_dutch_league.groupby('team_name')['form'].sum().reset_index()

pes_count_form = pes_dutch_league.groupby('team_name')['form'].count().reset_index(name='Total Players')

pes_sum_form['Average overall_rating'] = pes_sum_form['form']/pes_count_form['Total Players']

pes_sum_form['form']=pes_count_or['Total Players']

Dutch_League=pes_sum_form.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Dutch_League['team_name'], y=Dutch_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,255,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Form of Eredivisie League Team')

fig.show()