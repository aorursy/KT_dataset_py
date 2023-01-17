import numpy as np

import pandas as pd



from tqdm import tqdm_notebook as tqdm



GLOVE = '../input/glove840b300dtxt/glove.840B.300d.txt' #'../input/glove.840B.300d.txt'

LABELS = '../input/imet-2019-fgvc6/labels.csv'

TRAIN = '../input/imet-2019-fgvc6/train.csv'

TRAIN_IMG = '../input/imet-2019-fgvc6/train/{}.png'
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

all_words = ["abruzzi","achaemenid","aegean","afghan","after british","after german","after german original","after italian","after russian original","akkadian","alexandria-hadra","algerian","alsace","american","american or european","amsterdam","ansbach","antwerp","apulian","arabian","aragon","arica","asia minor","assyrian","atlantic watershed","attic","augsburg","augsburg decoration","augsburg original","austrian","avignon","avon","aztec","babylonian","babylonian or kassite","bactria-margiana archaeological complex","balinese","bavaria","bayreuth","beautiran","beauvais","belgian","berlin","birmingham","boeotian","bohemian","bologna","bordeaux","bow","brescia","bristol","british","british or french","british or scottish","brunswick","brussels","burma","burslem","byzantine","calima","cambodia","campanian","canaanite","canosan","castel durante","catalan","catalonia","caucasian","caughley","central asia","central european","central highlands","central italian","chalcidian","chantilly","chaumont-sur-loire","chelsea","chelsea-derby","chimu","china","chinese with dutch decoration","chinese with european decoration","chinese with french mounts","chiriqui","chorrera","chupicuaro","colima","colombian","colonial","colonial american","copenhagen","corinthian","coromandel coast","costa rica","costa rica or panama","cretan","crete","cyclades","cycladic","cypriot","cypriot or phoenician","czech","danish","deccan","dehua","delft","derby","deruta","devonshire","dresden","dublin","dutch","dyak","east greek","east greek/sardis","eastern european","eastern mediterranean","eastern mediterranean or italian","edinburgh","edomite","egypt","egyptian","elamite","england","etruria","etruscan","euboean","european","european bronze age","faliscan","ferrara","flemish","flemish or italian","florence","for american market","for british market","for continental market","for danish market","for european market","for french market","for iberian market","for portuguese market","for russian market","for swedish market","frankenthal","frankish","freiburg im breisgau","french","french or german","french or italian","french or swiss","fulda","furstenberg","gaul","geneva","genoa","german","german or swiss","ghassulian","gnathian","gonia","greek","greek islands","greek or roman","guanacaste-nicoya","gubbio","gurkha","haida","hanau","hattian","helladic","hilt","hittite","hochst","huastec","hungarian","hungary","huron","ica","inca","india","indian or nepalese","indonesia","inuit","iran","irish","isin-larsa","isin-larsaold babylonian","islamic","italian","italian or sicilian","italian or spanish","italic","italic-native","japan","javanese","jouy-en-josas","kathmandu valley","kazakhstan","kholmogory","kievan rus'","konigsberg","korea","la rochelle","laconian","lambayeque","lambeth","langobardic","leuven","lille","limoges","liverpool","london","london original","longton hall","lowestoft","ludwigsburg","lydian","lyons","macao","macaracas","macedonian","madrid","malayan","mali","manteno","maya","meissen","meissen with german","mennecy","mennecy or sceaux","mexican","mezcala","michoacan","milan","mimbres","minoan","mitanni","mixtec","moche","moche-wari","montelupo","moro","moroccan","moustiers","mughal","muisca","munich","mycenaean","nabataean","nailsea","nantes","naples","nasca","naxos","nayarit","neo-sumerian","neolithic","nepal","netherlandish","neuwied am rhein","nevers","nimes","north china","north indian","north italian","north netherlandish","northern european","northern india","northern italian","northwest china","northwest china/eastern central asia","norwegian","nuremberg","nymphenburg","old assyrian trading colony","olmec","orleans","ottonian","padua","pakistan","palermo","paracas","paris","parita","parthian","parthian or sasanian","peruvian","pesaro","philippine","phrygian","piedmont","polish","populonia","portuguese","potsdam","praenestine","proto-elamite","provincial","ptolemaic","qajar","quechua","remojadas","rhenish","roman","roman egyptian","rome","rouen","russian","saint-cloud","salinar","salzburg","san sabastian","sasanian","savoy","saxony","scandinavian","sceaux","scottish","scythian","seleucid","seville","sevres","sheffield","sicily","siena","silesia","sinceny","skyros","smyrna","south german","south italian","south netherlandish","southall","southern german","spanish","spitalfields","sri lankan","st. petersburg","staffordshire","stockholm","stoke-on-trent","strasbourg","sulawesi","sumatran","sumerian","surrey","swedish","swiss","syrian","tairona","tarentine","teano","teotihuacan","thailand","thanjavur","the hague","thessaly","thuringia","tibet","tibetan","tiwanaku","tlatilco","tlingit","tolita-tumaco","topara","tsimshian","turin","turkish","turkish or venice","ubaid","umbria","united states","unknown","urartian","urbino","urbino with gubbio luster","valencia","venice","veracruz","veraguas","verona","versailles","vienna","vietnam","villanovan","villeroy","vincennes","visigothic","vulci","wari","west slavic","western european","worcester","wurzburg","zenu","zoroastrian","zurich","abbies","abraham","abstraction","acanthus","acorns","acrobats","actors","actresses","adam","admirals","adonis","adoration of the magi","adoration of the sheperds","air transports","alexander the great","altars","amazons","amulets","amun","ancient greek","angels","anger","animals","anklet","annunciation","aphrodite","apocalypse","apollo","apostles","apples","arabic","archangel gabriel","arches","architects","architectural elements","architectural fragments","architecture","ariadne","armors","army","arrowheads","arrows","artemis","artists","assumption of the virgin","astronomy","athena","athletes","autumn","avalokiteshvara","axes","bacchus","badges","bagpipes","bakers","balconies","bamboo","baptism of christ","barns","baseball","basins","bathing","bathsheba","bats","battles","beaches","beads","beakers","bears","bedrooms","beds","bees","belts","benches","benjamin franklin","bes","bible","bicycles","billiards","birds","bishops","boars","boats","bobbins","bodhisattva","bodies of water","body parts","books","boots","bottles","bow and arrow","bowls","boxes","boxing","boys","bracelets","bridges","brooches","buckles","buddha","buddhism","buddhist religious figures","buffalos","buildings","buildings and structures","bulls","burial grounds","burials","butterflies","buttons","cabinets","calendars","camels","cameos","canals","candelabra","candles","candlesticks","cannons","capitals","carpets and rugs","carriages","cartouches","caryatids","castles","cathedrals","cats","cauldrons","caves","celestial bodies","censers","centaurs","ceremony","ceres","chairs","chalices","chariots","chess","chests","chickens","children","chinese","chinoiserie","christ","christian imagery","christianity","christmas","churches","circles","circus","cities","civil war","cleopatra","clocks","clothing and accessories","clouds","coat of arms","coats","coffeepots","coffins","coins","columns","commodes","concerts","contemplation","coptic","cornucopia","corpses","correspondence","corsets","costumes","couches","couples","courtyards","coverlets and quilts","cows","crabs","cradles","cranes","crescents","crocodiles","cross","crowd","crucifixion","cuneiform","cupid","cups","curtains","cutlery","daggers","daily life","daisies","dance","dancers","dancing","david","dawn","death","decorative designs","decorative elements","deer","deities","demons","descent from the cross","deserts","design elements","desks","devil","diadems","diamonds","diana","dice","dining","dionysus","dishes","docks","doctors","documents","dogs","dolls","dolphins","domes","donkeys","doors","doorways","doves","dragons","drawing","dresses","drinking","drinking glasses","drums","drunkenness","ducks","durga","eagles","earrings","easter","egg and dart","elephants","emblems","embroidery","emperor augustus","entombment","eros","esther","europa","eve","evening","ewers","eyes","facades","faces","factories","fairies","falcons","family","fans","farmers","farms","fathers","fauns","fear","feathers","feet","female nudes","fire","firearms","fireplaces","fireworks","fish","fishing","flags","flowers","flutes","fluting","food","footwear","forests","fortification","fountains","foxes","friezes","frogs","fruit","funerals","funerary objects","furniture","gadrooning","galatea","games","gardeners","gardens","garlands","gates","generals","genre scene","geometric patterns","george washington","gingham pattern","girls","globes","gloves","goats","goblets","goddess","gods","grapes","greek deities","greek figures","griffins","grotesques","guitars","hair","hammers","hands","harps","hathor","hats","hawks","heads","hell","helmets","hercules","hermes","hexagons","hieroglyphs","hills","hilts","hindu religious figures","hinduism","historical figures","holofernes","holy family","horns","horse riding","horses","horus","hospitals","houses","human figures","hunting","illness","incense burners","infants","inns","inscriptions","insects","insignia","interiors","isis","jackets","jainism","jars","jason","jesus","jewelry","jockeys","journals","judith","jugs","julius caesar","juno","jupiter","kettles","keys","kings","kitchens","knives","krishna","lace","ladders","ladles","lakes","lambs","lamentation","lamps","landforms","landscapes","last judgement","last supper","law","leaves","leda","leopards","lighting","lions","literature","liturgical objects","living rooms","lizards","lobsters","lockets","lotuses","louis xiv","love","lovers","lutes","madonna and child","maenads","magicians","maitreya","male nudes","mandolins","manjushri","manuscripts","maps","mark antony","markets","mars","mary magdalene","masks","massacres","medallions","medea","men","merchants","mercury","mice","military","military clothing","military equipment","minerva","mirrors","monkeys","monks","monsters","monuments","moon","moses","mosques","mothers","mountains","muses","music","musical instruments","musicians","mythical creatures","mythology","napoleon i","nativity","navy","necklaces","necktie","neptune","nero","netsuke","new testament","night","nike","nonrepresentational art","nymphs","obelisks","occupations","octagons","octopus","old testament","olive trees","opera","organs","ornament","orpheus","owls","painting","paisley","palaces","palmettes","pants","parks","parrots","party","peaches","peacocks","pediments","pendants","pentecost","peonies","percussion instruments","performance","perseus","pheasants","pianos","pigeons","pigs","pilasters","pinecones","pins","pitchers","plants","playing","playing cards","pocket watches","poetry","poets","polka-dot pattern","pomegranates","ponds","popes","portraits","poseidon","princes","princesses","prisms","prisoners","prisons","profiles","prostitutes","psyche","punishment","purses","putti","pyramids","queens","qur'an","rabbits","railways","rain","rams","reading","rectangles","religious events","religious texts","reliquaries","riding","rings","rivers","roads","robes","roman deities","roosters","rosaries","roses","rowing","ruins","sadness","sailors","saint anne","saint anthony","saint catherine","saint francis","saint george","saint jerome","saint john the baptist","saint john the evangelist","saint joseph","saint lawrence","saint mark","saint matthew","saint michael","saint paul","saint peter","saints","samples","sarcophagus","satire","satyrs","saucers","scarabs","scarves","schools","scorpions","screens","scrolls","sculpture","seals","seas","seascapes","seating furniture","self-portraits","serpents","servants","shakespeare","shakyamuni","sheep","shells","shepherds","shields","ships","shirts","shiva","shoes","sibyl","silenus","singers","singing","skeletons","skirts","skulls","sky","slavery","sleep","smoking","snails","snakes","snow","soldiers","spears","spectators","sphinx","sports","spring","squares","squirrels","stairs","stars","still life","stools","storage furniture","storms","strapwork","street scene","streets","stripes","students","suffering","suits","summer","sun","sundials","sunflowers","swans","sword guards","swords","tabernacles","tables","tablets","taoism","tapestries","taweret","tea caddy","tea drinking","teachers","teapots","telescopes","temples","tents","textile fragments","textiles","theatre","tigers","tombs","tools and equipment","towers","towns","toys","trains","transportation","trays","trees","triangles","tricorns","triton","trophies","trumpets","tulips","tunics","tureens","turtles","undergarment","uniforms","urns","utilitarian objects","vajrapani","vase fragments","vases","vegetables","venus","vestments","vests","victory","villages","vines","violas","violins","virgin mary","vishnu","volcanoes","vulcan","wagons","walking","wars","washing","watches","waterfalls","watermills","waves","weapons","weights and measures","wells","wind","windmills","windows","wine","winter","women","working","world war i","worshiping","wreaths","writing","writing implements","writing systems","zeus","zigzag pattern","zodiac"]
def build_matrix(all_words, path, statistics=True, embedding_index=None):

    if embedding_index is None: embedding_index = load_embeddings(path)

    

    if statistics:

        good_words = 0

        bad_words = 0

        for word in all_words:

            try:

                embedding_index[word]

                good_words = good_words+1

            except Exception as e:

                bad_words = bad_words+1

        print("good {}, bad {}, percent {}".format(good_words, bad_words, good_words/(good_words+bad_words)))

    

    embedding_matrix = dict()

    unknown_words = []

    text = ""

    

    for i, word in enumerate(all_words):

        try:

            embedding_matrix[word] = (np.array(embedding_index[word])).tolist()

        except KeyError:

            if " " in word or "-" in word:

                try:

                    words = word.replace("for ", "").replace(" or ", " ").replace(" and ", " ").replace(" with ", " ").replace(" of ", " ").replace("the ", " ").split(" ")

                    while "" in words:

                        words.remove("")

                    embedding_matrix[word] = (np.array([embedding_index[word] for word in words]).mean(axis=0)).tolist()

#                     text = text + "\n" + str(words)

                    continue

                except KeyError:

                    try:

                        words = (word.replace("for ", "").replace(" or ", " ").replace(" and ", " ").replace(" with ", " ").replace(" of ", " ").replace("the ", " ").replace("'", "").replace("/", " ") + "[]").replace("ish[]", "[]").replace("s[]", "[]").replace("[]", "").split(" ")

                        while "" in words:

                            words.remove("")

                        embedding_matrix[word] = (np.array([embedding_index[word] for word in words]).mean(axis=0)).tolist()

                        text = text + "\n" + str(words)

                        continue

                    except KeyError:

                        try:

                            words = (word.replace("for ", "").replace(" or ", " ").replace(" and ", " ").replace(" with ", " ").replace(" of ", " ").replace("the ", " ").replace("'", "").replace("/", " ") + "[]").replace("ish[]", "[]").replace("s[]", "[]").replace("[]", "").replace("-", " ").split(" ")

                            while "" in words:

                                words.remove("")

                            embedding_matrix[word] = (np.array([embedding_index[word] for word in words]).mean(axis=0)).tolist()

                            text = text + "\n" + str(words)

                            continue

                        except KeyError:

                                try:

                                    words = (word.replace("for ", "").replace(" or ", " ").replace(" and ", " ").replace(" with ", " ").replace(" of ", " ").replace("the ", " ").replace("'", "").replace("/", " ") + "[]").replace("ish[]", "[]").replace("s[]", "[]").replace("[]", "").replace("-", "").split(" ")

                                    while "" in words:

                                        words.remove("")

                                    embedding_matrix[word] = (np.array([embedding_index[word] for word in words]).mean(axis=0)).tolist()

                                    text = text + "\n" + str(words)

                                    continue

                                except KeyError:

                                    unknown_words.append(word)

                                    continue

                        continue

            elif "ish" in word or "s" or "''" in word:

                try:

                    words = (word+"[]").replace("ish[]", "[]").replace("s[]", "[]").replace("[]", "").replace("'", "").split(" ")

                    while "" in words:

                        words.remove("")

                    embedding_matrix[word] = (np.array([embedding_index[word] for word in words]).mean(axis=0)).tolist()

                    text = text + "\n" + str(words)

                    continue

                except KeyError:

                    unknown_words.append(word)

                    continue

            unknown_words.append(word)

    print(text)

    return embedding_matrix, unknown_words

embedding_index = load_embeddings(GLOVE)
embedding_matrix, unknown_words = build_matrix(all_words, GLOVE, embedding_index=embedding_index)

good_count = len(embedding_matrix.keys())

bad_count = len(unknown_words)

print("good {}, bad {}, percent {}".format(good_count, bad_count, good_count/(good_count + bad_count)))

# print("good {}, bad {}, percent {}".format(len(embedding_matrix.keys())-len(unknown_words), len(unknown_words), (len(embedding_matrix)-len(unknown_words))/(len(embedding_matrix)+len(unknown_words))))
unknown_words
embedding_index['quran']
from scipy.spatial.distance import cosine



print(cosine(embedding_index['english'], embedding_index['chinese']), cosine(embedding_index['english'], embedding_index['atom']))
import operator



def find_similar_word(target):

    dic = dict()

    if type(target) == str: target = embedding_matrix[target]

    for word in embedding_matrix.keys():

        dic[word] = cosine(target, embedding_matrix[word])

    return sorted(dic.items(), key=operator.itemgetter(1))
find_similar_word("female nudes")[:10] # well it is similar to virgin mary. The Russians love female nudes too!
print("Russian to Nudes: {}, Chinese to Nudes: {}".format(cosine(embedding_index["russian"], embedding_index["nudes"]), cosine(embedding_index["chinses"], embedding_index["nudes"])))
find_similar_word("european bronze age")
find_similar_word("zoroastrian")
# LABELS

labels = pd.read_csv(LABELS)

labels_attribute_id = [int(ids) for ids in labels.attribute_id]

labels_attribute_name = [name.replace("culture::", "").replace("tag::", "") for name in labels.attribute_name]

ids_2_names = dict(zip(list(labels_attribute_id), list(labels_attribute_name)))

names_2_ids = dict(zip(list(labels_attribute_name), list(labels_attribute_id)))



# TRAIN

train = pd.read_csv(TRAIN)

train_ids = train.id

train_attribute_ids = [   [int(i) for i in img_string.split(" ")]   for img_string in train.attribute_ids]

train_attribute_name = [   [ids_2_names[id]for id in ids]   for ids in train_attribute_ids]



# fake embeddings for NaN targets

for name in names_2_ids.keys():

    if name not in embedding_matrix.keys():

        embedding_matrix[name] = np.zeros(300).tolist()



train_attribute_embed = [   (np.array([embedding_matrix[name] for name in names]).sum(axis=0)/15).tolist()    for names in train_attribute_name]

train_attribute_name[:2]
train_attribute_embed = [   (np.array([embedding_matrix[name] for name in names]).sum(axis=0)/15).tolist()    for names in train_attribute_name]







# ids_2_embed = dict(zip(list(train_attribute_ids), list(train_attribute_embed)))

# train_attribute_name[:2], train_attribute_embed[:2]
import cv2



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



def find_similar_image(target, train_attribute_ids, train_attribute_embed):

    target_embed = train_attribute_embed[train_ids.values.tolist().index(target)]

    scores = []

    ids = []

    for i, embed in enumerate(train_attribute_embed):

        ids.append(train_ids[i])

        scores.append(cosine(embed, target_embed))

    dic = dict(zip(ids, scores))

    sort = sorted(dic.items(), key=operator.itemgetter(1))

    return dict(sort)



# def show_images(list_images_path):

#     from IPython.display import Image, display

#     for imageName in list_images_path:

#         display(Image(filename=imageName))



# credit: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

def show_images(list_images_path, cols = 1, titles = None):

    """Display a list of images in a single figure with matplotlib.

    

    Parameters

    ---------

    images: List of np.arrays compatible with plt.imshow.

    

    cols (Default = 1): Number of columns in figure (number of rows is 

                        set to np.ceil(n_images/float(cols))).

    

    titles: List of titles corresponding to each image. Must have

            the same length as titles.

    """

    images = [cv2.imread(img) for img in list_images_path]

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image)

        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.show()



# print(find_similar_image("1000483014d91860", train_attribute_ids, train_attribute_embed))

dic = find_similar_image("1000483014d91860", train_attribute_ids, train_attribute_embed)

show_images([TRAIN_IMG.format(img) for img in list(dic.keys())[:10]], cols=2, titles=list(dic.values())[:10])
show_images([TRAIN_IMG.format(img) for img in list(dic.keys())[10:20]], cols=2, titles=list(dic.values())[10:20])
dic = find_similar_image("101c8394ff6db02d", train_attribute_ids, train_attribute_embed)

print(list(dic.values())[:10])

show_images([TRAIN_IMG.format(img) for img in list(dic.keys())[:10]], cols=2, titles=list(dic.values())[:10])