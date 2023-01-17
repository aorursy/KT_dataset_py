# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import sys

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
datafiles = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        ifile = os.path.join(dirname, filename)
        if ifile.split(".")[-1] == "json":
            datafiles.append(ifile)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install Pattern3
!sed -i "36s/.*/    pass/" /opt/conda/lib/python3.7/site-packages/pattern3/text/tree.py

from collections import Counter
import matplotlib.pyplot as plt
from pattern3.en import pluralize, singularize

class Paragraph_info:
    
    def __init__(self):
        self.corona_nams_list = list()
        self.count_corona_name = 0
        self.count_animal = 0


class Animal_info:
    
    def __init__(self):
        self.overall_count = 0
        self.paragraph_count = 0
        self.paragraph_info_dic = {}
animal_list =['raccoon Dog', 'citrinella', 'Sciurus niger', 'Ring-tailed lemur', 'chimpanzee', 'burmese black mountain Tortoise', 'Tauraco porphyrelophus',
               'Felis libyca', 'three-banded Plover', 'river Wallaby', 'Common brushtail possum', 'alpaca', 'chilean Flamingo', 'Greater rhea', 'ostriches',
                'mongooses', 'eudyptes', 'Pine siskin', 'Speothos vanaticus', 'Malleefowl', 'herring Gull', 'egyptian Goose', 'american Badger', 'Feral rock pigeon',
               'Salmon pink bird eater tarantula', 'Spheniscus mendiculus', 'african Polecat', 'gastropod', 'sportive Lemur', 'swans', 'Nyctereutes procyonoides',
               'Coracias caudata', 'Flightless cormorant', 'Helmeted guinea fowl', 'pterodroma', 'Long-tailed skua', 'Knob-nosed goose', 'phoca', 'Graspus graspus', 'bornean',
               'Common langur', 'red-tailed Phascogale', 'vespertilio', 'Manouria emys', 'siiiquosa', 'Southern boubou', 'Flying fox', 'hepatozoon', 'Slender-billed cockatoo',
               'microplitis', 'weevil', 'Anas bahamensis', 'Catharacta skua', 'strap', 'crab-eating Raccoon', 'Horned rattlesnake', 'greenfinches', 'Pycnonotus nigricans',
               'american Woodcock', 'black Kite', 'Bison bison', 'Oriental short-clawed otter', 'ornithorhynchus', 'common Seal', 'arenaria', 'naevi', 'black and white Colobus',
               'long-billed Cockatoo', 'herpestes', 'Crown of thorns starfish', 'Galapagos hawk', 'Bos mutus', 'American black bear', 'feathering', 'Crested porcupine',
               'finch', 'snake', 'tiger', 'blue Duck', 'white-throated Toucan', 'Bassariscus astutus', 'Roseat flamingo', 'ridden', 'Toddy cat', 'Macropus rufogriseus', 'storks',
               'red-necked Phalarope', 'Black-winged stilt', 'bleeding heart Monkey', 'Woodrat', 'Greater adjutant stork', 'green-backed Heron', 'Crocuta crocuta',
               'Crocodylus niloticus', 'resurgencia', 'Macropus eugenii', 'Red-billed toucan', 'gibraltar', 'leucolysatcs', 'weasels', 'Litrocranius walleri', 'procambarus',
               'European spoonbill', 'Hippotragus equinus', 'Taurotagus oryx', 'flies', 'niviventer', 'american virginia Opossum', 'Western bearded dragon', 'ringtail Cat', 'Ninox superciliaris',
               'wallaby', 'Ara macao', 'anguilla', 'Passer domesticus', 'shelducks', 'four-striped grass Mouse', 'Black-tailed deer', 'brown Hyena', 'gannets', 'grivet',
               'emerald green tree Boa', 'brolga Crane', 'red-billed buffalo Weaver', 'Trumpeter swan', 'Galapagos albatross', 'railway', 'scary', 'Carduelis uropygialis',
               'glaber', 'Deroptyus accipitrinus', 'granti', 'beetle', 'cules', 'Microcebus murinus', 'Panthera leo', 'snails', 'foxes', 'Ardea cinerea', 'Procyon cancrivorus', 'Geochelone radiata',
               'indian tree Pie', 'Phasianus colchicus', 'rabbits', 'Hymenolaimus malacorhynchus', 'eudorcas', 'bobcats', 'Snake', 'Llama', 'clam', 'sungorus', 'fowl', 'flea',
               'Crotalus triseriatus', 'Gray rhea', 'Francolinus leucoscepus', 'Martes americana', 'brown and yellow Marshbird', 'Cordylus giganteus', 'macrobrachium', 'earthworms',
               'masked Booby', 'Painted stork', 'pelicans', 'black Curlew', 'Brush-tailed rat kangaroo', 'Certotrichas paena', 'bactrian', 'Eastern box turtle',
               'Australian pelican', 'antibovine', 'Vanessa indica', 'hoary Marmot', 'blattella', 'Felis wiedi or Leopardus weidi', 'dolphins',
               'harpiocephalus', 'white-necked Raven', 'Fairy penguin', 'crocuta', 'greater Roadrunner', 'lobsters', 'Civet', 'planctomycetes', 'red-breasted Cockatoo', 'Dromedary camel',
               'Irania gutteralis', 'Yellow-throated sandgrouse', 'erinaceus', 'larvae', 'north american Beaver', 'Phacochoerus aethiopus', 'oncomelania', 'western lowland Gorilla',
               'Hystrix cristata', 'Greater blue-eared starling', 'hare', 'painted Stork', 'typicum', 'mesobuthus', 'Southern hairy-nosed wombat', 'Larus novaehollandiae', 'pulchellus',
               'goldfish', "swainson's Francolin", 'panthera', 'mites', 'Greater sage grouse', 'anatinus', 'goliath Heron', 'Eremophila alpestris', 'Chlidonias leucopterus',
               'swan', 'Lorythaixoides concolor', 'Common melba finch', 'eastern cottontail Rabbit', 'Asian openbill', 'pig-tailed Macaque', 'Masked booby',
               'North American red fox', 'Warthog', 'white-rumped Vulture', 'gray Langur', 'beaks', 'Gyps bengalensis', 'European badger', 'mastiff', 'Connochaetus taurinus',
               'steller sea Lion', 'Varanus sp.', 'Lepus arcticus', 'long-nosed Bandicoot', 'Bubo virginianus', 'Kaffir cat', 'mite', 'Anser caerulescens',
               'Aquila chrysaetos', 'Vanellus chilensis', 'Black-eyed bulbul', 'butterflies', 'cercopithecus', 'turkey', 'Xerus sp.', 'Spotted wood sandpiper',
               'mothur', 'european Shelduck', 'Frilled lizard', 'glossy Ibis', 'okosongoro', 'Bettongia penicillata', 'indian Mynah', 'hippoboscid', 'Little grebe', 'robin',
               'Madagascar fruit bat', 'tree Porcupine', 'Black-cheeked waxbill', 'Northern phalarope', 'pedunculatum', 'brown Antechinus', 'asian water Dragon', 'Bengal vulture',
               'Pronghorn', 'meleagris', 'Rock dove', 'ovis', 'Honey badger', 'Leopard', 'royal Tern', 'Gyps fulvus', 'leopardus', 'banded Mongoose', 'american Buffalo',
               'African wild cat', 'Aegypius tracheliotus', 'roseat Flamingo', 'African snake', 'Pampa gray fox', 'Common wallaroo', 'canada Goose', 'Asian lion',
               'purple Moorhen', 'Felis silvestris lybica', 'crown of thorns Starfish', 'freshwater', 'Varanus salvator', 'Bugeranus caruncalatus', 'serrulatus', 'Black-backed jackal',
               'Common raccoon', 'Darwin ground finch', 'palmatus', 'Vine snake', 'shrews', 'monkeys', 'termites', 'Anhinga rufa', 'Geospiza sp.', 'Boa constrictor mexicana',
               'black-cheeked Waxbill', 'iratimadhuram', 'sockeye Salmon', 'grey Heron', 'plains Zebra', 'bald Eagle', 'Waved albatross', 'kristin', 'sauria', 'lactalbumin', 'elizabeth',
               'Paradoxurus hermaphroditus', 'Phylurus milli', 'Golden brush-tailed possum', 'Blue-breasted cordon bleu', 'pollinators', 'asian Elephant', 'sage Grouse',
               'White-cheeked pintail', 'Tawny eagle', 'Vulpes cinereoargenteus', 'Grus rubicundus', 'Macropus giganteus', 'African lion', 'Glossy ibis',
               'australian brush Turkey', 'bateleur Eagle', 'Orcinus orca', 'belangeri', 'superb Starling', 'asian Openbill', 'corvus', 'idella', 'sarsr', 'capuchin',
               'Mycteria leucocephala', 'latrans', 'Palm squirrel', 'African lynx', 'two-banded Monitor', 'Pied crow', 'Cape wild cat', 'Dasypus septemcincus', 'capers',
               'Ephippiorhynchus mycteria', 'sambar', 'bovines', 'Seiurus aurocapillus', 'Otocyon megalotis', 'sylvaticus', 'sandhill Crane', 'Phalacrocorax albiventer', 'Praying mantis',
               'peafowl', 'canaries', 'Wild turkey', 'minks', 'Bleeding heart monkey', 'Proteles cristatus', 'Banded mongoose', 'purple Grenadier', 'orchards', 'Mazama americana',
               'gorilla', 'Western lowland gorilla', 'Laughing dove', 'Red and blue macaw', 'ruditapes', 'water Monitor', 'Blue-tongued lizard', 'Sauromalus obesus', 'pinnipedia',
               'european Beaver', 'Hummingbird', 'Tsessebe', 'Crax sp.', 'mexicanus', 'Great cormorant', 'ostrea', 'gaii', 'octogon', 'Cracticus nigroagularis', 'Indian red admiral',
               'Capreolus capreolus', 'camelid', 'Paroaria gularis', 'Kobus defassa', 'parasitoid', 'amberjack', 'Prionace glauca', 'australian Pelican', 'Ploceus rubiginosus', 'Impala',
               'Tetracerus quadricornis', 'shiny', 'forget', 'procyonoides', 'grevyi', 'heron', 'Ceratotherium simum', 'convoluta', 'brush-tailed rat Kangaroo', 'American marten', 'goila',
               'Bradypus tridactylus', 'uinta ground Squirrel', 'pygoscelis', 'hrpconjugated', 'occidentalis', 'Numbat', 'vervet', 'yellow-billed Stork', 'Timber wolf',
               'eurasian Hoopoe', 'Mirounga leonina', 'hispidus', 'gopherus', 'Common shelduck', 'albatross', 'martens', 'eastern dwarf Mongoose', 'mallard', 'spit',
               'lepidopteran', 'Cormorant', 'acacia', 'Spermophilus parryii', 'geese', 'Epicrates cenchria maurus', 'Amphibolurus barbatus', 'black-footed Ferret',
               'mourning collared Dove', 'giant Otter', 'harbor Seal', 'Striped hyena', 'Chionis alba', 'Falco peregrinus', 'Choloepus hoffmani', 'shellfish', 'bears',
               'guanaco', 'moschata', 'Killer whale', 'Lophoaetus occipitalis', 'Tasmanian devil', 'greylag Goose', 'tammar Wallaby', 'bonobos', 'mormopterus', 'Brazilian tapir',
               'Dendrocygna viduata', 'paddocks', 'mosquito', 'katingan', 'hereford', 'Tringa glareola', 'mellifera', 'lybica', 'Red-billed tropic bird',
               'kanni', 'eastern diamondback Rattlesnake', 'north american red Fox', 'Manatee', 'Cape Barren goose', 'shrimp', 'tortoises', 'Sceloporus magister', 'aphids',
               'Nelson ground squirrel', 'Emerald green tree boa', 'egyptian Vulture', 'leopards', 'freeranging', "verreaux's Sifaka", 'Red-shouldered glossy starling', 'timber Wolf',
               'Chital', 'blue-tongued Skink', 'galapagos Albatross', 'Blue and yellow macaw', 'auratus', 'african clawless Otter', 'common Eland', 'Gray langur',
               'Charadrius tricollaris', 'Ictonyx striatus', 'maggots', 'kelp Gull', 'native Cat', 'lepus', 'Colobus guerza', 'constitute', 'loaves', 'daubentonii', 'loggerhead',
               'Lepus townsendii', 'Crowned eagle', 'Hudsonian godwit', 'Adouri', 'crab', 'hupensis', 'indian Peacock', 'sika', 'Mephitis mephitis',
               'pine Siskin', 'African polecat', 'ruddy', 'monotreme', 'black-capped Capuchin', 'Dark-winged trumpeter', 'tyrant Flycatcher', 'reclusa',
               'long-tailed spotted Cat', 'Pelecanus occidentalis', 'hyacinthinus', 'whiptail', 'capybara', 'Mountain duck', 'cape White-eye', 'anas', 'Ovibos moschatus',
               'psittacus', 'Pocket gopher', 'partridge', 'gibbons', 'angora', 'carpet Python', 'patas', 'reindeer', 'buzzing', 'bactrianus', 'Gaur', 'Cape clawless otter', 'cape Raven',
               'Tachybaptus ruficollis', 'australian sea Lion', 'Snowy sheathbill', 'gazelles', 'Ciconia episcopus', 'Oriental white-backed vulture', 'Fringe-eared oryx',
               'crocodylus', 'Varanus komodensis', 'snowy Egret', 'guttata', 'Cercopithecus aethiops','Asian elephant', 'Brown and yellow marshbird', 'yellow-billed Hornbill',
               'koala', 'pigeons', 'coprologiques', 'feijo', 'bohor Reedbuck', 'Mocking cliffchat', 'lizards', 'Tiliqua scincoides', 'Galapagos mockingbird', 'Red-tailed hawk',
               'Hydrochoerus hydrochaeris', 'african Porcupine', 'American alligator', 'canids', 'Gray duiker', 'Four-horned antelope',
               'Tapirus terrestris', 'American bison', 'Wild water buffalo', 'serotinus', 'Northern fur seal', "Grant's gazelle", 'Goliath heron',
               'Common ringtail', 'Chuckwalla', 'Equus burchelli', 'White-throated kingfisher', 'Dicrostonyx groenlandicus', 'Rose-ringed parakeet', 'Dasypus novemcinctus',
               'goddess', 'red Hartebeest', 'Leptoptilos crumeniferus', 'papio', 'geoffroyi', 'Mississippi alligator',
               'Carduelis pinus', 'red Sheep', 'Cygnus atratus', 'Whip-tailed wallaby', 'pholidota', 'Delphinus delphis', 'Culex', 'blue Racer', 'Desert tortoise',
               'eastern fox Squirrel', "pallas's fish Eagle", 'Little brown bat', 'Golden-mantled ground squirrel', 'cockatoos', 'Macropus fuliginosus',
               'Hippotragus niger', 'nematode', 'palaemon', 'budgerigar', 'Bos taurus', 'Dusky gull', 'australian spiny Anteater', 'eastern boa Constrictor', 'Redunca redunca',
               'Cheetah', 'terriers', 'javan gold-spotted Mongoose', 'elegant crested Tinamou', 'ortomyxoviridae', 'pediculus', 'Worm snake', 'golden Jackal', 'Agkistrodon piscivorus',
               'Eastern quoll', 'Otaria flavescens', 'Perameles nasuta', 'marsupial', 'snake-necked Turtle', 'Red-tailed wambenger', "Clark's nutcracker", 'lagotis', 'Egyptian viper',
               'Burmese black mountain tortoise', 'Genoveva', 'red Phalarope', 'White-throated monitor', 'macaque', 'Prairie falcon', 'Ixodes', 'bearded', 'Eagle owl',
               'Meerkat', 'Common zebra', 'rotacoronavirus', 'buffaloes', 'Violet-crested turaco', 'quadricornis', 'gerbil', 'green-winged Macaw',
               'Short-beaked echidna', 'White-browed sparrow weaver', 'Tenrec ecaudatus', 'Water monitor', 'Ferruginous hawk', 'Lesser masked weaver', 'Macaca fuscata', 'red howler Monkey',
               'vipers', 'haloxylon', 'Southern tamandua', 'Tiger cat', 'tokay Gecko', 'haemaphysalis', 'leucopus', 'Larus fuliginosus', 'Malachite kingfisher', 'Brush-tailed bettong',
               'Butterfly', 'kobus', 'Cochlearius cochlearius', 'Bobcat', 'Columba palumbus', 'alces', 'black Vulture',
               'Lesser flamingo', 'Black kite', 'gibbon', 'Antelope ground squirrel', 'infests', 'rockhopper', 'rainbow Lory', 'salamanders', 'Spermophilus armatus','Alpaca', 'beavers',
               'Southern white-crowned shrike', 'Phalacrocorax brasilianus', 'guanacos', 'Ursus americanus', 'Rufous tree pie', 'silver-backed Jackal', 'Black bear',
               'alternaria', 'Red brocket', 'millivora', 'sharks', 'rhinoceros', 'Red deer', 'qatar', 'common Rhea', 'Goanna lizard', 'Laniarius ferrugineus', 'Meleagris gallopavo',
               'nycticebus', 'camels', 'striped Hyena', 'rats', 'Dromaeus novaehollandiae', 'adelie', 'Geococcyx californianus', 'hyaena', 'ectoparasitic', 'cat', 'cape Fox',
               'mule Deer', 'comb Duck', 'Water legaan', 'clemmys', "Levaillant's barbet", 'indian Leopard', 'Rufous-collared sparrow',
               'Mara', 'humanus', 'Tropical buckeye butterfly', 'Griffon vulture', 'brown Lemur', 'Bushpig', 'rufous', "Bennett's wallaby", 'armadillos', 'ibex', 'Recurvirostra avosetta',
               'White-fronted capuchin', 'erithacus', 'great Kiskadee', 'Uraeginthus angolensis', 'buzz', 'african Lion', 'Red phalarope', 'Baleen whale', 'bonobo',
               'Brown pelican', 'Square-lipped rhinoceros', 'antelopes', 'Sidewinder', 'Gerenuk', 'great horned Owl', 'snowshoe',
               'yellow-rumped Siskin', 'tryon', 'wildcat', 'cristata', 'Canadian river otter', 'canescens', 'white Spoonbill', "Swainson's francolin", 'Columbian rainbow boa', 'Paca',
               'fitcconjugated', 'Purple moorhen', 'Greater flamingo', 'Ploceus intermedius', 'black-throated Cardinal', 'Thamnolaea cinnmomeiventris', 'daboia', 'Corallus hortulanus cooki',
               'Komodo dragon', 'Streptopelia senegalensis', 'phalacrocorax', 'Barrows goldeneye', 'Limosa haemastica', 'tilapia', 'Barking gecko', 'common Genet',
               'Monitor Legaa', 'cerrasco', 'European wild cat', 'Malabar squirrel', 'Urial', 'Red-breasted nuthatch', 'murchronata', 'hircus', 'calidris',
               'white-fronted Bee-eater', 'Pelecans onocratalus', 'Blue wildebeest', 'Chestnut weaver', 'hamsters', 'American racer',
               "Richardson's ground squirrel", "miner's Cat", 'Stanley crane', 'Blue waxbill', 'dusky Gull', 'Ringtail', 'african Jacana', 'pampa gray Fox', 'quinquestriatus', 'miracidia',
               'Woolly-necked stork', 'euro Wallaby', 'red-legged Pademelon', 'galapagos Hawk', 'larval', 'sandpipers', "Denham's bustard", 'south african Hedgehog',
               'Canis lupus', 'chinook', 'Agama sp.', 'pale-throated three-toed Sloth', 'porcine', 'white-mantled Colobus', 'black Bear', 'wallabies', 'large-eared Bushbaby',
               'Brown capuchin', 'White-fronted bee-eater', 'komodo Dragon', 'crested Barbet', 'Salvadora hexalepis', 'tiger Snake', 'jagged', 'garter', 'batinfaf',
               'Alcelaphus buselaphus caama', 'meriones', 'Kelp gull', 'pied Kingfisher', 'Cynomys ludovicianus', 'african Skink', 'Zenaida asiatica', 'Spectacled caiman',
               "Kirk's dik dik", 'long-billed Corella', 'egyptian Cobra', 'dipterans', 'Haliaeetus leucocephalus', 'grebes', 'gelada Baboon',
               'Sciurus vulgaris', 'Bush dog', 'Brazilian otter', 'Hanuman langur', 'Cathartes aura', 'digitaria', 'hawk', 'greater Rhea', 'Lamprotornis chalybaeus',
               'Lesser mouse lemur', 'rotundus', 'Gray heron', 'hottentot Teal', 'arctic ground Squirrel', 'Eastern fox squirrel', 'Koala',
               'Podargus strigoides', 'melanoleuca', 'Bubo sp.', 'Callipepla gambelii', 'alligators', 'chimp', 'crab-eating Fox', 'black-throated butcher Bird',
               'drebot', 'larvas', 'fat-tailed Dunnart', 'Canis dingo', 'helvum', 'Plains zebra', 'little brown Bat', 'Dendrohyrax brucel', 'silver-haired', 'african Darter', 'Feline',
               'White-rumped vulture', 'Madagascar hawk owl', 'Tailless tenrec', 'beagles', 'white-throated Monitor', 'Oribi', 'Rhesus monkey', 'fallow',
               'myomorph', 'Nesomimus trifasciatus', 'collared Lemming', 'Ibex', 'Callorhinus ursinus', 'phocoena', 'common Duiker', 'Panthera tigris',
               'Radiated tortoise', 'two-toed tree Sloth', 'mississippi Alligator', 'rhesus Monkey', 'monodelphis', 'eurasian Beaver', 'tropical buckeye Butterfly',
               'Lamprotornis nitens', 'common Ringtail', 'mosquitoes', 'european wild Cat', 'chestnut Weaver', 'Chordeiles minor', 'maned', 'jumper', 'fairy Penguin', 'mississipi',
               'Zonotrichia capensis', 'Herring gull', 'Chilean flamingo', 'Marine iguana', 'Trichoglossus haematodus moluccanus', 'Macaca radiata', 'common Nighthawk', 'striatus',
               'Ictalurus furcatus', 'pongo', 'haemolysates', 'Red-breasted cockatoo', 'stanley Bustard', 'edulis', 'Sambar', 'American crow',
               'long-finned pilot Whale', 'Dicrurus adsimilis', 'Acanthaster planci', 'Platypus', 'woolly-necked Stork', 'Eurocephalus anguitimens', 'Macropus rufus',
               'unguiculatus', 'aethiops', 'Buttermilk snake', 'shorebirds', 'green Heron', 'toads', 'Lesser double-collared sunbird', 'Agouti', 'canadian river Otter',
               'sarus Crane', 'Two-toed sloth', 'Black-capped capuchin', 'sigmodon', 'Ephipplorhynchus senegalensis', 'Monitor lizard', 'sagamianum',
               'Ringtail cat', 'carp', 'Zorilla', 'burmese brown mountain Tortoise', 'rat', 'feline', 'Oreamnos americanus', 'wolf', 'North American porcupine', 'Galictis vittata',
               'Indian tree pie', 'common Raccoon', 'eutherian', 'mandrill', 'Gymnorhina tibicen', 'calliphora', 'whale', 'Sylvicapra grimma',
               'heterometrus', 'frogs', 'Propithecus verreauxi', 'caribou', 'spelaea', 'wild Turkey', 'Macaca mulatta', 'toddy Cat', 'parrot',
               'lions', 'Megaderma spasma', 'Bent-toed gecko', 'Turtle', 'Bubalus arnee', 'Pelecanus conspicillatus', 'Hoary marmot', 'zebra',
               'Silver-backed jackal', 'Great white pelican', 'Roe deer', 'yellow Mongoose', 'osbom', 'Tammar wallaby', 'Osprey',
               'white-browed Owl', 'white-necked Stork', 'turtles', 'Red kangaroo', 'Common long-nosed armadillo', 'tortoise', 'horse', 'argentatus', 'Cercatetus concinnus',
               'Phalacrocorax carbo', 'beluga', 'Tamandua tetradactyla', 'naped', 'Sylvilagus floridanus', 'White-necked raven', 'Australian spiny anteater', 'Blackish oystercatcher',
               'Savannah deer', 'piglets', 'gnathostomes', 'Common dolphin', 'rhesus Macaque', 'snakeroot', 'Tachyglossus aculeatus', 'Papio ursinus', 'American beaver', 'Felis concolor',
               'tammar', 'Ovis musimon', 'Greater roadrunner', 'Grey lourie', 'swallow-tail Gull', 'Seven-banded armadillo', 'Crab-eating fox', 'sphinx', 'thirteen-lined Squirrel',
               'Crab-eating raccoon', 'Frog', 'Rhea americana', 'dinan', 'Eurasian beaver', 'pangolins', 'Galapagos dove', 'Thylogale stigmatica',
               'American bighorn sheep', 'Black-tailed prairie dog', 'Raphicerus campestris', 'Saddle-billed stork', 'camelids', 'dammah', 'penguin',
               'Crowned hawk-eagle', 'tasmanian Devil', 'mandras tree Shrew', 'colchicus', 'common palm Civet', 'salmo', 'Nine-banded armadillo', 'pumas', 'jubatus',
               'blue-tongued Lizard', 'Yellow baboon', 'ponies', 'Castor fiber', 'Felis chaus', 'flukes', 'Limnocorax flavirostra', 'common Waterbuck',
               'Agile wallaby', 'bighorn', 'frameviral', 'Phalaropus lobatus', 'Common wolf', 'Striped skunk', 'asiatic Jackal', 'Bucorvus leadbeateri', 'turkey Vulture',
               'Arctic tern', 'Topi', 'Nyctanassa violacea', 'Phascolarctos cinereus', 'Common waterbuck', 'carassius', 'squirrel Glider', 'carpet Snake',
               'River wallaby', 'Downy woodpecker', 'common brushtail Possum', 'yellow-necked Spurfowl', 'djungarian', 'Green heron', 'african black Crake', 'Arboral spiny rat',
               'Long-tailed jaeger', 'brown Pelican', 'House sparrow', 'eothenomys', 'Dasyprocta leporina', 'Sarcophilus harrisii', 'Tree porcupine',
               'African ground squirrel', 'south american Puma', 'magnificent frigate Bird', 'white-fronted Capuchin', 'lutra', 'Blacksmith plover', 'american Racer',
               'marshalli', 'southern Screamer', 'Mandras tree shrew', 'Alouatta seniculus', 'Brown lemur', 'blue Fox', 'Malagasy ground boa', 'Black vulture',
               'White spoonbill', 'California sea lion', 'petrel', 'cygnus', 'pyhin', 'lava Gull', 'silver-backed Fox', 'kori Bustard', 'galv', 'Common turkey',
               'Elegant crested tinamou', 'Black-throated butcher bird', 'gull', 'fulmars', 'Turkey vulture',
               'black-backed Magpie', 'white-throated Robin', 'Eunectes sp.', 'owls', 'Giant heron', 'suncus', 'Ciconia ciconia', 'Grizzly bear', 'Canis lupus baileyi',
               'brush-tailed Phascogale', 'mercenaria', 'bufo', 'Onager', 'cereopsis Goose', 'Petaurus breviceps', 'Uinta ground squirrel', 'red Meerkat',
               'two-toed Sloth', 'raider', 'Coendou prehensilis', 'Blue catfish', 'Bald eagle', 'Scarlet macaw', 'Gelada baboon', 'Milvago chimachima',
               'red Brocket', 'grizzly Bear', 'berkhoffii', 'yellow-bellied Marmot', 'amur', 'parvovirosis', 'bod', 'baldeschwieler', 'Great skua', 'land Iguana', 'loons',
               'Lepilemur rufescens', 'pastures', 'tupaia', 'greater sage Grouse', 'coimbatore', 'Indian leopard', 'indian giant Squirrel', 'owsianka', 'fork-tailed Drongo',
               'hisat', 'plicata', 'green vine Snake', 'little Heron', 'religiosa', "Leadbeateri's ground hornbill", 'Pied kingfisher',
               'dromedaries', 'Field flicker', 'Burmese brown mountain tortoise', 'chloris', 'Glossy starling', 'black-capped Chickadee', 'Australian brush turkey',
               'Melanerpes erythrocephalus', 'acanthias', 'Ostrich', 'white-faced tree Rat', 'Plectopterus gambensis', 'saanen',
               'antelope', 'Mungos mungo', 'koi', 'waterfowls', 'Spotted deer', 'macacus', 'Least chipmunk', 'capreolus', 'Bos frontalis', 'Red squirrel',
               'golden-mantled ground Squirrel', 'crane', 'montana', 'Gila monster', 'Western palm tanager', 'Acinynox jubatus', 'musk', 'Aedes',
               'Threskionis aethiopicus', 'orectolobus', 'King vulture', 'Groundhog', 'Woylie', 'Carpet python', 'Blue-faced booby', 'Anitibyx armatus', 'Eolophus roseicapillus', 'Wolf spider', 'cattle Egret',
               'Naja haje', 'ovine', 'northern Phalarope', "Hoffman's sloth", 'collared Lizard', 'Macropus parryi', 'microtus', 'Three-banded plover', 'sand', 'Choriotis kori',
               'White-winged dove', 'Australian magpie', 'Bontebok', 'Yellow-rumped siskin', 'warmblood', 'Silver gull', 'House crow', 'curve-billed Thrasher',
               'northern elephant Seal', 'White-tailed jackrabbit', 'Potoroo', 'chinchillas', 'anopheles', 'Dusicyon thous', 'otters', 'Phalacrocorax varius', 'malagasy ground Boa',
               'Coluber constrictor foxii', 'Caiman crocodilus', 'large Cormorant', 'Western grey kangaroo', 'Suricata suricatta', 'rufus', 'western spotted Skunk', 'Sportive lemur',
               'myrmecia', 'ford', "bennett's Wallaby", 'Pseudoleistes virescens', 'Oncorhynchus nerka', 'Chameleon', 'Cebus apella', 'esox', 'tuna',
               'short-nosed Bandicoot', 'kafue flats Lechwe', 'Mexican beaded lizard', 'murid', 'laniger', 'sulfur-crested Cockatoo', 'Asiatic wild ass', 'pygmy', 'culicoides', 'South African hedgehog',
               'goat', 'Cuis', 'Common green iguana', 'Pine squirrel', 'rutilus', 'sloth Bear', 'roelke', 'pied Cormorant', 'crabs', 'arctic Tern', 'cape barren Goose',
               'Tockus erythrorhyncus', 'pecten', 'Scolopax minor', 'Little heron', 'Cape white-eye', 'arvicola', 'Aonyx cinerea', 'Blue-footed booby', 'horned Puffin', 'Corvus albus', 'Hystrix indica',
               'Branta canadensis', 'nubian Bee-eater', 'Red-headed woodpecker', 'duck', 'Ara ararauna', 'threw', 'penicillidia', 'african Elephant', 'plecotus', 'White stork', 'cattle',
               'Psittacula krameri', 'Lasiorhinus latifrons', 'galapagos Penguin', 'Nasua nasua', 'leiurus', 'golden Eagle', 'Carphophis sp.', 'Tadorna tadorna',
               'Centrocercus urophasianus', 'Upupa epops', 'Eleven-banded armadillo', 'Pale white-eye', 'Canis lupus lycaon', 'Coqui partridge', 'Macaca nemestrina', 'Pedetes capensis',
               'Alcelaphus buselaphus cokii', 'subscription', 'chukar', 'Kobus vardonii vardoni', 'llama', "Smith's bush squirrel", 'aselliscus', 'Spilogale gracilis',
               'blue Shark', 'Red hartebeest', 'Buteo regalis', 'Crotalus cerastes', 'Anastomus oscitans', 'Ceryle rudis', 'Pytilia melba', 'Gerbil', 'Reindeer', 'Cacatua tenuirostris',
               'dog', 'tsetse', 'Nasua narica', 'Spur-winged goose', 'malabar Squirrel', 'boat-billed Heron', "hoffman's Sloth", 'Cervus elaphus', 'grey-footed Squirrel',
               'Southern right whale', 'spectacled', 'nilssonii', 'urinating', 'African jacana', 'Collared lemming', 'racoon', 'Tyto novaehollandiae', 'Oreotragus oreotragus', 'Fisher',
               'Mule deer', 'gouldian', 'Dipodomys deserti', 'Yellow-bellied marmot', 'wistar', 'murine', 'raccoon', 'african wild Dog', 'Cottonmouth', 'Laughing kookaburra', 'african fish Eagle',
               'African red-eyed bulbul', 'Brindled gnu', 'lemur', 'bugs', 'Ground monitor', 'strix', 'Russian dragonfly', 'Lama guanicoe', 'eastern white Pelican', 'savanna Fox',
               'Steller sea lion', 'Erinaceus frontalis', 'spotted wood Sandpiper', 'dahlstrom', 'allodermanyssus', 'Short-nosed bandicoot', 'Haliaetus vocifer', 'Cobra', 'common boubou Shrike',
               'greater blue-eared Starling', 'sociable Weaver', 'foina', 'Theropithecus gelada', 'flounder', 'Black rhinoceros', 'seven-banded Armadillo', 'North American beaver',
               'American buffalo', 'Small Indian mongoose', 'goats', 'Boat-billed heron', 'Giant girdled lizard', 'wattled Crane', 'White-mantled colobus', 'tursiops', 'hydatidiform',
               'Spermophilus tridecemlineatus', 'Greater kudu', 'Phlebotomus papatasii', 'mynah', 'silver Gull', 'Carmine bee-eater', 'scorpions', 'Damaliscus dorcas', 'feral rock Pigeon',
               'jackal', 'Felis rufus', 'Lybius torquatus', 'southern Lapwing', 'Australian sea lion', 'African black crake', 'pied', 'goanna Lizard', 'gazella',
               'cheetahs', 'cervids', 'white-tailed Deer', 'Crotaphytus collaris', 'prehensile-tailed Porcupine', 'Damaliscus lunatus', 'Marmota monax', 'chimpanzees', 'trichechus',
               'Cyrtodactylus louisiadensis', 'Pseudocheirus peregrinus', 'Egyptian vulture', 'galapagos Mockingbird', 'Australian masked owl', 'Ornate rock dragon', 'cormorants', 'taurinus',
               'vulpes', 'Margay', 'australian Magpie', 'Northern elephant seal', 'purpureus', 'turtle', 'Marabou stork', 'Crab', 'Petaurus norfolcensis', 'Black-necked stork',
               'Puma', 'Cebus albifrons', 'Arctic hare', 'red-headed Woodpecker', 'grey Lourie', 'Jungle kangaroo', 'herbaceous',
               'arboral spiny Rat', 'white-winged black Tern', 'Thirteen-lined squirrel', 'Myotis lucifugus', 'Grey mouse lemur', 'blue Catfish', 'Equus hemionus', 'magistrate black Colobus',
               'Little brown dove', 'tailless Tenrec', 'Galah', 'Roseate cockatoo', 'rufous tree Pie', 'Amazon parrot', 'Lappet-faced vulture', 'crayfish', 'Cape cobra',
               'western patch-nosed Snake', 'white-tailed Jackrabbit', 'tigers', 'hamster', 'Common pheasant', 'ferret', 'Tyrant flycatcher',
               'yellow-headed Caracara', 'porcupines', 'malachite Kingfisher', 'Haliaetus leucogaster', 'gagcaataac', 'freight', 'southern sea Lion', 'amsaa', 'house Sparrow', 'ducks', 'killer Whale',
               'red-necked Wallaby', 'red Deer', 'hawks', 'sarstedt', 'Leptoptilus dubius', "Gambel's quail", 'palm Squirrel', 'burrowing', 'Giant armadillo', 'Aedes albopictus', 'dogfish',
               'Tiger snake', 'Black swan', 'lepidoptera', 'Nannopterum harrisi', 'indian star Tortoise', 'blue and yellow Macaw', 'Western pygmy possum', 'asian false vampire Bat',
               'puzachenko', 'american Bison', 'campo Flicker', 'Eastern boa constrictor', 'Ursus arctos', 'Kongoni', 'White-nosed coatimundi', 'Canada goose', 'collared Peccary', 'Lamprotornis superbus',
               'Boa caninus', 'Canis latrans', 'Woodchuck', 'lockhart', 'gazelle', 'spinus', 'camelidae', 'African skink', 'indian Jackal', 'Two-banded monitor', 'Javanese cormorant', 'lanigera',
               'Yellow-crowned night heron', 'Thalasseus maximus', 'swine', 'Heloderma horridum', 'snakehead', 'Hyalomma', 'auritus', 'giant girdled Lizard', 'megalobrema',
               'Red sheep', 'lorises', 'Japanese macaque', 'rousettus', 'roe Deer', 'Common nighthawk', 'Swallow-tail gull', 'laughing Kookaburra', 'prawns', 'Martes pennanti',
               'slender Loris', 'tropicbirds', 'Herpestes javanicus', 'Cygnus buccinator', 'Black-fronted bulbul', 'piglet', 'schloegel', 'Anser anser', 'blue-footed Booby', 'brindled Gnu',
               'Alopochen aegyptiacus', 'chinstrap', 'red Squirrel', 'Libellula quadrimaculata', "cook's tree Boa", 'Snow goose', 'Neotoma sp.', 'Green vine snake', 'Tawny frogmouth', 'badgers',
               'Struthio camelus', 'crowned Hawk-eagle', 'glama', 'pelican', 'nigripes', 'eagles', 'Common seal', 'rattlesnake', 'Common genet', 'Echimys chrysurus', 'snowy Owl',
               'common Langur', 'Lamprotornis sp.', 'chachalacas', 'Old world fruit bat', 'opossum', 'llamas', 'Trichosurus vulpecula', 'Cacatua galerita', 'pheasant', 'Olive baboon', 'lobster', 'japonensis',
               'South American meadowlark', 'sphyraena', 'Crotalus adamanteus', 'fulmar', 'Tayassu tajacu', 'litura', 'Potamochoerus porcus', 'Spermophilus lateralis', 'musk Ox',
               'African bush squirrel', 'cullinectes', 'Gerbillus sp.', 'Vombatus ursinus', 'Toxostoma curvirostre', 'Long-necked turtle', 'crested Bunting', 'Eastern indigo snake',
               'Macropus agilis', 'sheep', 'chicken', 'Gazella thompsonii', 'wagtail', 'Crested bunting', 'Serval', 'Mellivora capensis', 'Neotis denhami', 'Long-tailed spotted cat',
               'radiated Tortoise', 'Naja sp.', 'coelacanth', "Pallas's fish eagle", 'Sloth bear', 'earle', 'barasingha Deer', 'Dingo', 'king Vulture', 'platypus', 'mykiss',
               'whip-tailed Wallaby', 'musophagiformes', 'tigris', 'Vanellus sp.', 'openbill Stork', 'nathusii', 'Lava gull', "grant's Gazelle", 'Spotted-tailed quoll',
               'Black-crowned crane', 'Sarus crane', 'gray Duiker', 'Gekko gecko', 'sprague', 'spotted Deer', 'sciurus', 'white Stork', 'Procyon lotor', 'siberian', 'gray Rhea',
               'eastern grey Kangaroo', 'chipmunk', 'water Moccasin', 'sapelovirus', 'Alces alces', 'Canadian tiger swallowtail butterfly', 'Mourning collared dove', 'guenon', 'crocodiles', 'barnacle',
               'anemone', 'southern white-crowned Shrike', 'Bubalornis niger', 'nudix', 'boulant', 'Rainbow lory', 'Eudromia elegans', 'stints', 'Pituophis melanaleucus', 'Cape fox',
               'black-necked Stork', 'albifrons', 'common Goldeneye', 'Sus scrofa', 'mussels', 'moehlman', 'Hottentot teal', "Verreaux's sifaka", 'Indian jackal', 'southern Boubou',
               'Grey-footed squirrel', 'menhaden', 'korros', 'mule', 'Plocepasser mahali', 'rangifer', 'Lily trotter', 'Helogale undulata', 'weeper Capuchin', 'horned',
               'common Wombat', 'pelecanus', 'cape wild Cat', 'blue Waxbill', 'discosoma', 'African wild dog', 'Trachyphonus vaillantii', 'Sockeye salmon', 'cottontails',
               'Red-tailed cockatoo', 'Loris tardigratus', 'Antechinus flavipes', 'African darter', 'alectoris', 'black-collared Barbet', 'African buffalo', 'Large-eared bushbaby', 'hippo',
               'guanicoe', 'south american sea Lion', 'square-lipped Rhinoceros', 'wolves', 'barking Gecko', 'Grison', 'Odocoilenaus virginianus', 'Brown brocket', 'yacare', 'african Buffalo',
               'adustus', 'horses', 'roan Antelope', 'Cervus duvauceli', 'larus', 'Defassa waterbuck', 'White-tailed deer', 'Giant anteater', 'Black curlew', 'gerbils',
               'Antidorcas marsupialis', 'diptera', 'grey mouse Lemur', 'gulls', 'hen', 'Superb starling', 'turdus', 'palaemonetes', 'crested Screamer',
               'dark-winged Trumpeter', 'White-browed owl', 'White-headed vulture', 'Tragelaphus angasi', 'Falco mexicanus', 'Bushbuck', 'Ardea golieth', 'four-spotted Skimmer', 'Collared peccary',
               'Chlamydosaurus kingii', 'ayuune', 'hamadryas', 'crocodile', 'clams', 'Bahama pintail', 'musca', 'worms', 'Anas platyrhynchos', 'lemmings',
               'Asian false vampire bat', 'Haematopus ater', "levaillant's Barbet", 'Hyrax', 'ferruginous Hawk', 'tawny Frogmouth', 'sanguineus', 'diaemus', 'eels', 'Magellanic penguin',
               'Fork-tailed drongo', 'Sandhill crane', 'Neophron percnopterus', 'capensis', 'Nyctea scandiaca', 'motacilla', 'beisa Oryx', 'Striated heron', 'southern brown Bandicoot',
               'golden brush-tailed Possum', 'Chloephaga melanoptera', 'house Crow', 'Eastern diamondback rattlesnake', 'hippoglossus', 'Common mynah', 'porpoises', "denham's Bustard",
               'orangutans', 'mainka', 'blackish Oystercatcher', 'ring-tailed Possum', 'Anthropoides paradisea', 'Sulfur-crested cockatoo', 'Crested barbet', 'white-winged Tern', 'greater Kudu',
               'apodemus', 'urochordata', 'Pygmy possum', 'Notechis semmiannulatus', 'red-capped Cardinal', 'Mouflon', 'anopheline', 'lilac-breasted Roller',
               'Geochelone elephantopus', 'Rangifer tarandus', 'pine Squirrel', 'bombus', 'tapirus', 'Marmota caligata', 'mexican Boa', 'aegypti',
               'desert kangaroo Rat', 'Dolichitus patagonum', 'Moose', 'Sacred ibis', 'common long-nosed Armadillo', 'bahama Pintail', 'Merops bullockoides', 'Phaethon aethereus',
               'Indian porcupine', 'polar Bear', 'Corvus albicollis', 'indian Porcupine', 'Long-billed cockatoo', 'gese', 'dama Wallaby', 'olive Baboon', 'marabou Stork', 'genets',
               'Loxodonta africana', 'badger', 'eurasian Badger', 'Egyptian goose', 'tachyglossus', 'epomophorus', 'whitefish', 'partridges', 'red-billed Toucan', 'red-billed Hornbill',
               'oriental short-clawed Otter', 'dairy', 'Turtur chalcospilos', 'mustelus', 'frilled Dragon', 'Smithopsis crassicaudata', 'South American sea lion', 'glandarius', 'Platalea leucordia',
               'Conolophus subcristatus', 'Swamp deer', 'water Legaan', 'sandflies', 'Pine snake', 'giraffe', 'Galapagos penguin', 'White-throated toucan', 'merchandising', 'Zosterops pallidus',
               'Himantopus himantopus', 'Didelphis virginiana', 'japanese Macaque', 'Acrobates pygmaeus', 'Eira barbata', 'Lizard', 'Mycteria ibis', 'common Wallaroo', 'Canis aureus',
               'Ara chloroptera', 'European shelduck', 'violet-crested Turaco', 'Steenbok', 'skunks', 'Ground legaan', 'wood Pigeon', 'Red-billed buffalo weaver', 'california sea Lion',
               'Melophus lathami', 'asian foreset Tortoise', "steller's sea Lion", 'dromedary Camel', 'european Spoonbill',
               'Diomedea irrorata', 'rhea', 'mouse', 'Red-legged pademelon', 'Vanellus armatus', 'Desert spiny lizard', 'woolly', 'oysters', 'turnstones', 'wobbegong',
               'lily Trotter', 'hudsonian Godwit', 'Yellow-billed hornbill', 'White-lipped peccary', 'Plegadis falcinellus', 'Panthera onca', 'Kalahari scrub robin',
               'Golden eagle', 'knob-nosed Goose', 'buttermilk Snake', 'monkey', 'rodent', 'Vulpes chama', 'african pied Wagtail', 'odocoileus', 'eastern indigo Snake', 'Mountain goat',
               'Ring-tailed gecko', 'Jabiru stork', 'tibrogargan', 'Hippopotamus', 'pandalus', 'Acridotheres tristis', 'Indian peacock', 'black Swan', 'Eastern dwarf mongoose', 'Wapiti Elk',
               'Arctic lemming', 'mlinaric', 'Comb duck', 'drosophila', 'common Mynah', 'Stanley bustard', 'Nubian bee-eater', 'Weeper capuchin', 'cervus', 'neotropic Cormorant', 'long-tailed Skua',
               'baboons', 'Dama wallaby', 'Rhabdomys pumilio', 'blacksmith Plover', 'Stone sheep', 'White-winged black tern',
               'Striped dolphin', 'Larus sp.', 'russian Dragonfly', 'Jaguarundi', 'Cereopsis novaehollandiae', 'guinea pig', 'long-crested hawk Eagle', 'bobwhite', 'Lama glama', 'squirrel',
               'Yellow-billed stork', 'muntjac', 'ground Legaan', 'prawn', 'rhipicephalus', 'siskins', 'skunk', 'scotomanes', 'White-bellied sea eagle', 'elk Wapiti', 'lizard',
               'Phalaropus fulicarius', 'Galapagos sea lion', 'cow', 'bubalus', 'hardwickii', 'Black and white colobus', 'cats', 'redlegged',
               'Speotyte cuniculata', 'Phascogale calura', 'defassa Waterbuck', 'wings', 'western grey Kangaroo', 'lentivial', 'Sagittarius serpentarius', 'Black spider monkey', 'common melba Finch',
               'wild water Buffalo', 'black-crowned night Heron', 'mol', 'dusky Rattlesnake', 'red lava Crab', 'Colaptes campestroides', 'emerald-spotted wood Dove',
               'Lasiodora parahybana', 'Phalacrocorax niger', 'Blesbok', 'snow Goose', 'Junonia genoveua', 'Anaconda', 'saccolaimus', 'South American puma', 'laughing Dove',
               'European red squirrel', 'Guanaco', 'burrowing Owl', 'jerboa', 'Yellow-headed caracara', 'Giant otter', 'Wood pigeon',
               'Bohor reedbuck', 'white-headed Vulture', 'scottish highland Cow', 'standardbred', 'Indian star tortoise', 'Naja nivea', 'Water moccasin', 'raccoons', 'jungle Kangaroo',
               'Paddy heron', 'magellanic Penguin', 'Great horned owl', 'bonnet Macaque', 'shorthaired', 'boobies', 'australian masked Owl', 'galapagos sea Lion', 'marmosets', 'Neotropic cormorant',
               'Black-crowned night heron', 'Tockus flavirostris', 'Streptopelia decipiens', 'pallas', 'Nile crocodile', 'pbst', 'Marmota flaviventris', 'Cynictis penicillata', 'Kafue flats lechwe',
               'emus', 'panleukopenia', 'Cereopsis goose', 'arctic Hare', 'Capybara', 'frilled Lizard', 'Brush-tailed phascogale', 'Canis mesomelas', 'Great egret',
               'ring-tailed Gecko', 'terrestris', 'trout', 'Asian water dragon', 'fish', 'Plegadis ridgwayi', 'Frilled dragon', 'Pterocles gutturalis', 'Feathertail glider', 'caballus',
               'sparrow', 'Savanna baboon', 'elaphus', 'little Cormorant', 'coqui Francolin', 'Stenella coeruleoalba', 'sable Antelope', 'niloticus', 'Panthera pardus',
               'Eastern white pelican', 'Sable antelope', 'melomys', 'Asiatic jackal', 'Steenbuck', 'greater adjutant Stork', 'Snake-necked turtle', 'caduelis',
               'pollens', 'Grey heron', 'Oryx gazella callotis', 'ferrets', 'Burrowing owl', 'Caracal', 'dipteran', 'desert Tortoise',
               'hyaenas', 'sally lightfoot Crab', 'Harbor seal', 'doves', 'antelope ground Squirrel', 'Antilocapra americana', 'sorex', 'prairie Falcon', 'ambrosia', 'Agama lizard',
               'penguins', 'ailuropoda', 'Nucifraga columbiana', 'Curve-billed thrasher', 'platyrhynchos', 'microsporum', 'Common zorro', 'casserole', 'Snowy owl', 'dander',
               'Dassie', 'wasps', 'spur-winged Goose', 'bipedal', 'Pacific gull', 'spiders', 'Blue-tongued skink', 'Red howler monkey', 'bison', 'asian Lion', 'roseate Cockatoo', 'bats',
               'violet-eared Waxbill', 'American badger', 'Squirrel glider', 'maculatus', 'Felis caracal', 'lesser mouse Lemur', 'Butorides striatus', 'swamp Deer',
               'hyporthodus', 'Anathana ellioti', 'coccinella', 'canadensis', 'Common wombat', 'rufous-collared Sparrow', 'sandpiper', 'North American river otter', 'chickens',
               'Asian foreset tortoise', 'phacochoerus', 'great Cormorant', 'black Rhinoceros', 'pteropus', 'stanley Crane', 'clupeidae', 'black-winged Stilt', 'scarlet Macaw', 'King cormorant',
               'marine Iguana', 'Sugar glider', 'beef', 'lamarckii', 'Panthera leo persica', 'Spotted hyena', 'kaffir Cat', 'Pitangus sulphuratus', 'alpacas', 'Francolinus coqui', 'seals',
               'Phascogale tapoatafa', 'Picoides pubescens', 'Erethizon dorsatum', 'Yellow-brown sungazer', 'horned Lark', 'aequorea', 'African pied wagtail', 'Little cormorant', 'pied butcher Bird',
               'Chauna torquata', 'eastern Quoll', 'rattus', 'Bee-eater', 'Green-winged macaw', 'bovine', 'jackals', 'Orca', 'Violet-eared waxbill', 'canine', 'Pteropus rufus',
               'Globicephala melas', 'rubripes', 'Mountain lion', 'elaphe', 'Southern lapwing', 'red-tailed Cockatoo', 'Alectura lathami', 'Eastern cottontail rabbit',
               'Asian water buffalo', 'basecalled','lemurs', 'Sociable weaver', 'pictus', 'american black Bear', 'Tayassu pecari', 'prairie',
               'Common palm civet', 'savanna Baboon', 'oriental white-backed Vulture', 'lipped', 'Grey phalarope', 'meerkats', 'Amazona sp.', 'helmeted guinea Fowl', 'blackbird', 'carrion',
               'Trichechus inunguis', 'four-horned Antelope', 'Haliaeetus leucoryphus', 'insectivore', 'chick', 'alan', 'firetail', 'small-spotted Genet',
               'african wild Cat', 'darter', 'Mustela nigripes', 'quails', 'aurita', 'Pseudalopex gymnocercus', 'pimephales', 'quail', 'Psophia viridis', 'carolinensis', 'Francolinus swainsonii',
               'curlew', 'Long-nosed bandicoot', 'gymnogongrus', 'cormorant', 'tarandus', 'muskoxen', 'short-beaked Echidna', 'asian water Buffalo',
               'pacos', 'red and blue Macaw', 'frigatebirds', 'Snowy egret', 'Horned puffin', 'Andean goose', 'cape Cobra', 'Carpet snake', 'pavo', 'Red-knobbed coot', 'bent-toed Gecko',
               'Argalis', 'common Zebra', 'Ursus maritimus', 'common Pheasant', 'american Alligator', 'Drymarchon corias couperi', 'weddell', 'Arctic ground squirrel', 'wildebeest',
               'white Rhinoceros', 'Eastern grey kangaroo', 'toad', 'Red-necked wallaby', 'culex', 'pygmy Possum', 'African clawless otter', 'iberian', 'degu', 'pufferfish',
               'mosquitos', 'black-backed Jackal', 'Arctic fox', 'Axis axis', 'Red-capped cardinal', 'ticks', 'nine-banded Armadillo', 'Red meerkat', 'Bottle-nose dolphin', 'vison', 'laevis', 'Merops sp.',
               'Calyptorhynchus magnificus', 'western bearded Dragon', 'cercocebus', 'sugar Glider', 'swordfish', 'African elephant', 'Javan gold-spotted mongoose', 'White rhinoceros', 'pied Crow',
               'Southern sea lion', 'redbelly', 'Macropus robustus', 'Fat-tailed dunnart', 'field Flicker', 'recluse', 'trutta', 'Aepyceros mylampus', "thomson's Gazelle", 'Pandon haliaetus',
               'conus', 'eurasian red Squirrel', 'aegyptiacus', 'least Chipmunk', 'Fulica cristata', 'Gopherus agassizii', 'fish', 'squalus', 'stramineus', 'Semnopithecus entellus', 'brown Brocket',
               'cyanobacterium', 'boars', 'Lemur fulvus', 'sage Hen', 'guereza', 'Ratufa indica', 'Lama pacos', 'sphaerophoria', 'Common boubou shrike', 'turnstone', 'Asian red fox',
               'expectorate', 'kalahari scrub Robin', 'Mabuya spilogaster', 'striated Heron', "richardson's ground Squirrel", 'lama', 'Tarantula', 'moles', 'nimbus',
               'blue-breasted cordon Bleu', 'vultures', 'ornithodoros', 'Southern ground hornbill', 'boeing', "azara's Zorro", 'Myrmecophaga tridactyla', 'Black-tailed tree creeper', 'Yak', 'Sea birds',
               'red-cheeked cordon Bleu', 'flowerpiercer', 'Sun gazer', 'blue Crane', 'dromedarius', 'Red-tailed phascogale', 'Collared lizard', 'Odocoileus hemionus', 'ostrich',
               'common Turkey', 'Cervus canadensis', 'albatrosses', 'red-tailed Hawk', 'opossums', 'Kobus leche robertsi', 'black-tailed Deer', 'Uraeginthus bengalus', 'tinamous',
               'Ovis dalli stonei', 'moose', 'pardus', 'Jungle cat', 'Two-toed tree sloth', 'grey Phalarope', 'red Kangaroo', 'Elephas maximus bengalensis', 'cubs',
               'Black-capped chickadee', 'Pied avocet', 'scotophilus', 'Acrantophis madagascariensis', 'Casmerodius albus', 'mice',
               'southern elephant Seal', 'Phoenicopterus ruber', 'Ring-tailed coatimundi', 'peromyscus', 'squirrels', 'mormoopidae', 'Southern brown bandicoot', 'Cabassous sp.', 'leopard',
               'mesocricetus', 'Green-backed heron', 'crimson-breasted Shrike', 'gorillas', 'Larus dominicanus', 'Kinkajou', 'African porcupine', 'dermacentor', 'Corvus brachyrhynchos',
               'Small-clawed otter', 'Geochelone elegans', 'insidiosus', "Burchell's gonolek", 'phlebotomus', 'Campo flicker', 'Pan troglodytes', 'pteronyssinus',
               'Ant', 'Microcavia australis', 'hemionus', 'Eutamias minimus', 'Corythornis cristata', 'Anopheles', 'sparrows', 'Cape starling', 'Prehensile-tailed porcupine', 'Cebus nigrivittatus',
               'Red-billed hornbill', 'red-billed tropic Bird', 'Ring dove', 'Aegypius occipitalis', 'african red-eyed Bulbul', 'okapis', 'Blue and gold macaw', 'Stick insect', 'ornate rock Dragon',
               'indian red Admiral', 'north american river Otter', 'Milvus migrans', '', 'thoroughbred', 'blue and gold Macaw', 'Magistrate black colobus', 'Felis yagouaroundi', 'Lapwing', 'martes',
               'javanese Cormorant', 'aepyceros', 'Suricate', 'Agelaius phoeniceus', 'Pig-tailed macaque', 'Trichoglossus chlorolepidotus', 'desert spiny Lizard',
               'Antilope cervicapra', 'tiger Cat', 'Gazella granti', 'arctic Fox', 'cane', 'whitetailed', 'Phoenicopterus chilensis', 'american bighorn Sheep', 'Pycnonotus barbatus',
               'Papilio canadensis', 'rheas', 'arctic Lemming', 'andean Goose', 'Capra ibex', 'Great kiskadee', 'common green Iguana', 'humpback', 'styliferus', 'Bare-faced go away bird',
               'Vicugna vicugna', 'Huron', 'naja', 'Sterna paradisaea', 'pig', 'Sally lightfoot crab', 'fowls', "leadbeateri's ground Hornbill", 'Blue crane', 'nelson ground Squirrel',
               'Small-spotted genet', 'giant Heron', 'Buteo jamaicensis', 'multimammate', 'little blue Penguin', 'Tursiops truncatus', 'prairie dogs', 'Silver-backed fox',
               'small-toothed palm Civet', 'White-necked stork', 'white-throated Kingfisher', 'bengal Vulture', 'Currasow', 'Nectarinia chalybea', 'rosmarus', 'white-winged Dove',
               'Yellow-necked spurfowl', 'caprine', 'Hawk-headed parrot', 'Giraffe', 'Coluber constrictor', 'little brown Dove', 'red-breasted Nuthatch', 'Coqui francolin', 'White-faced tree rat',
               'snowy Sheathbill', 'southern Tamandua', 'spotted Hyena', 'barrows Goldeneye', 'Pteronura brasiliensis', 'tick', 'Iguana iguana', 'Blue fox',
               'gray Heron', 'Royal tern', 'salmon pink bird eater Tarantula', 'greater Flamingo', 'Ammospermophilus nelsoni', 'cockles', 'Tragelaphus strepsiceros',
               "Steller's sea lion", 'cape clawless Otter', 'Camelus dromedarius', 'common Shelduck', 'scaly-breasted Lorikeet', 'american Crow', 'Gemsbok', 'lappet-faced Vulture',
               'griffon Vulture', 'crassostrea', 'southern ground Hornbill', 'giant Armadillo', 'Python', 'oplegnathus', 'yellow Baboon', 'Estrilda erythronotos',
               'Grus antigone', 'Priodontes maximus', 'orangutan', 'Tragelaphus scriptus', "Azara's zorro", 'northern fur Seal', "coke's Hartebeest", 'shrimps', 'european red Squirrel',
               'Pavo cristatus', 'white-cheeked Pintail', "smith's bush Squirrel", 'Felis pardalis', 'crocidura', 'hemiechinus', 'Amblyrhynchus cristatus', 'turkeys', 'Long-billed corella', 'Polar bear',
               'Malay squirrel', 'jaguars', 'greylag', 'Nycticorax nycticorax', 'Leprocaulinus vipera', 'Brolga crane', 'Papio cynocephalus', 'Horned lark', 'Cervus unicolor', 'bactrians',
               'Vulpes vulpes', 'Neophoca cinerea', 'Mirounga angustirostris', 'lotor', 'egypt', 'Bubulcus ibis', 'connochaetes', 'Morelia spilotes variegata', 'long-necked Turtle',
               'Myrmecobius fasciatus', 'spiny', 'Anas punctata', 'Yellow mongoose', 'Eurasian badger', 'phasianus', 'Bird', 'Cougar', 'chlorocebus', 'racer Snake', 'diplopylidium',
               'kangourous', 'brazilian Tapir', 'suricatta', 'budgerigars', 'Cattle egret', 'egyptian Viper', 'Springhare', 'salmon', 'Spheniscus magellanicus', "clark's Nutcracker", 'rabbit',
               'black-eyed Bulbul', 'Musk ox', 'Grey fox', 'black-crowned Crane', 'Coyote', 'yellow-brown Sungazer', 'white-bellied sea Eagle', 'marmoset',
               'Galago crassicaudataus', 'moth', 'Phoeniconaias minor', 'chacma Baboon', 'Mazama gouazoubira', 'emperor', 'Black-collared barbet', 'Red-necked phalarope', 'Ring-tailed possum',
               'madagascar fruit Bat', 'Dasyurus viverrinus', 'common Dolphin', 'immunoresearch', 'beaked', 'Creagrus furcatus', 'lesser Flamingo', 'American woodcock', 'Openbill stork',
               'columbian rainbow Boa', 'blue-faced Booby', 'peregrine Falcon', 'streptopelia', 'Blue peacock', 'sumatran', 'crustaceans', 'snub', 'Dendrocitta vagabunda', 'cow', 'stamen',
               'Peregrine falcon', 'parrots', 'phocine', 'Ateles paniscus', 'Tamiasciurus hudsonicus', 'Ctenophorus ornatus', 'troglodytes', 'Secretary bird', 'Dusky rattlesnake', 'Isoodon obesulus',
               'Scottish highland cow', 'Agouti paca', 'jararaca', 'madagascar hawk Owl', 'Bonnet macaque', 'giant Anteater', 'ring-tailed Coatimundi', 'equid', 'Bucephala clangula',
               'aardwolf', 'hadrurin', 'Gecko', 'pale White-eye', 'sun Gazer', 'goose', 'galapagos Dove', 'asiatic wild Ass', "burchell's Gonolek", 'Kori bustard',
               'muscovy', 'Boselaphus tragocamelus', 'bottlenose', 'Sula dactylatra', 'Mallard', 'Gorilla gorilla', 'wild Boar', 'koehler', 'microcebus', 'Eubalaena australis', 'Motacilla aguimp',
               'feathertail Glider', 'bat-eared Fox', 'oryceropus', 'long-tailed Jaeger', 'Tokay gecko', 'Sarcorhamphus papa', 'weasel', 'vitulina', 'steelhead',
               'rose-ringed Parakeet', 'odobenus', 'african bush Squirrel', 'gibbum', 'Lutra canadensis', 'Ovis ammon', 'Zalophus californicus', 'guineafowl', 'White-throated robin',
               'ramosissima', 'white-faced whistling Duck', 'mamiya', 'Diceros bicornis', 'Dabchick', 'oncorhynchus', "Cook's tree boa", 'waved Albatross', 'rerio',
               'dianleter', 'Rana sp.', 'viridis', 'Tayra', 'Western spotted skunk', 'Egyptian cobra', 'Chacma baboon', 'striped Skunk', 'Genetta genetta', 'Mexican boa', 'Sage hen', 'benato',
               'Long-finned pilot whale', 'cestoda', 'Grus canadensis', 'Climacteris melanura', 'red-knobbed Coot', 'loxodonta', 'galapagos Tortoise', 'little Grebe',
               'Lycaon pictus', 'Madoqua kirkii', 'microbat', 'Castor canadensis', 'Pale-throated three-toed sloth', 'Beisa oryx', 'Wattled crane', 'perspicillata', 'Blue racer', 'Aardwolf',
               'Ramphastos tucanus', 'cottontail', 'Ornithorhynchus anatinus', 'Potos flavus', 'Common eland', 'pied Avocet', 'bare-faced go away Bird', 'asian red Fox', 'sarslike', 'Guerza',
               'Oxybelis fulgidus', 'Small-toothed palm civet', 'canadian tiger swallowtail Butterfly', 'camelus', 'Four-striped grass mouse',
               'mexican Wolf', 'murinus', 'maniculatus', 'nyctereutes', 'saddle-billed Stork', 'Roan antelope', 'Cape raven', 'great Egret', 'Wild boar', 'bluetongue', 'aedes',
               'trumpeter Swan', 'Merops nubicus', 'taeniopygia', 'concolor', 'rock Dove', 'thoroughbreds', 'Lilac-breasted roller', 'Western patch-nosed snake',
               'Phoca vitulina', 'great Skua', 'mountain Goat', 'flightless Cormorant', 'Dasyurus maculatus', 'Ovenbird', 'Bat-eared fox', 'crowned Eagle', 'secretary Bird',
               'black-faced Kangaroo', 'striped Dolphin', 'Eumetopias jubatus', 'Raccoon dog', 'hanuman Langur', 'north american Porcupine', 'Long-crested hawk eagle',
               'Myiarchus tuberculifer', 'krill', 'flava', 'hawk-headed Parrot', 'Caribou', 'Ourebia ourebi', 'southern hairy-nosed Wombat', 'Alligator mississippiensis', 'eucampsipoda', 'jugend',
               'colobus', 'snakes', 'Fregata magnificans', 'Eurasian red squirrel', 'common Zorro', 'lesser double-collared Sunbird', 'lagopus',
               'Chelodina longicollis', 'red-tailed Wambenger', 'mocking Cliffchat', 'bottle-nose Dolphin', 'pacific Gull', 'downy Woodpecker', 'aotus', 'Red lava crab', 'coyotes', 'beaver',
               'Ovis orientalis', 'cyprinus', 'Nilgai', 'manis', 'vicugna', 'civet Cat', 'white-lipped Peccary', 'red-shouldered glossy Starling', 'Eudyptula minor', 'baleen Whale', 'mallards',
               'Civet cat', 'Varanus albigularis', 'alligator', 'pyrrhophyta', 'yellow-crowned night Heron', 'gentoo', 'Euro wallaby', 'Desert kangaroo rat',
               'rattlesnakes', 'small-clawed Otter', 'Lycosa godeffroyi', 'european Badger', 'mongoose', 'cape Starling', 'purpuratus', 'Brown antechinus', 'Galapagos tortoise',
               'black spider Monkey', 'Black-backed magpie', 'lice', 'sturgeon', 'red-winged Blackbird', 'porcupine', 'kitten', 'Spizaetus coronatus', 'Snycerus caffer', 'Lemur catta',
               'american Beaver', 'yellow-throated Sandgrouse', 'Spermophilus richardsonii', "Thomson's gazelle", 'Springbok', 'Stercorarius longicausus',
               'Oxybelis sp.', 'Nyala', 'white-browed sparrow Weaver', 'Jaguar', 'coqui Partridge', 'Crimson-breasted shrike', 'goldfinches', 'brazilian Otter', 'mountain Duck',
               'Oryx gazella', 'jabiru Stork', 'mexican beaded Lizard', 'aethiopicus', 'Ovis canadensis', 'Philetairus socius', 'Large cormorant', 'strepsiceros', 'Indian mynah', 'reinhardtii',
               'Meles meles', 'Sitta canadensis', 'deer', 'hematophagous', 'ferociously', 'Rhesus macaque', 'Physignathus cocincinus', 'hyenas', 'Potorous tridactylus', 'Leipoa ocellata', 'mephitis',
               'fulmarus', 'Common duiker', 'bush Dog', 'Mexican wolf', 'Caracara', 'horned Rattlesnake', 'mongolian', 'copepods', 'small indian Mongoose', 'dromedary',
               'brown Capuchin', 'Four-spotted skimmer', 'ring-tailed Lemur', 'pandas', 'american Marten', 'bivalve', 'acanthaster', 'Southern screamer', 'Red-winged hawk', 'green-winged Trumpeter',
               'Funambulus pennati', 'Pied cormorant', 'andersoni', 'farinae', 'bees', 'pigeon', 'glycannetviewer', 'fringe-eared Oryx', 'phyllostomus',
               'Melursus ursinus', 'spotted-tailed Quoll', 'Scaly-breasted lorikeet', "kirk's dik Dik", 'stone Sheep', 'Puku', 'Land iguana', 'Slender loris',
               'bubalis', 'cytisus', 'waterbuck', 'Red-cheeked cordon bleu', 'Zenaida galapagoensis', 'african Lynx', 'savannah Deer', 'echidna', 'lion', 'paracamelus', 'Sarkidornis melanotos',
               'Puna ibis', 'Dacelo novaeguineae', 'ring-necked Pheasant', 'nile Crocodile', 'Little blue penguin', 'Blue shark', 'Parus atricapillus', 'Pied butcher bird',
               'Barasingha deer', 'Bateleur eagle', 'eastern box Turtle', 'spectacled Caiman', 'Racer snake', 'Aonyx capensis', 'Felis serval', 'fikrig',
               'lonnbergi', 'slender-billed Cockatoo', "Miner's cat", 'western pygmy Possum', 'mangabey', 'Black-throated cardinal', 'Blackbuck', 'Fratercula corniculata',
               'pheasants', 'Gabianus pacificus', 'Magnificent frigate bird', 'dogs', 'Green-winged trumpeter', 'Chamaelo sp.', 'great white Pelican', 'Brown hyena', 'ring Dove',
               'European stork', 'baboon', 'cynocephalus', 'melogale', 'mountain Lion', 'grey Fox', 'gallopavo', 'mink', 'Red-winged blackbird', 'cobra', 'hedgehogs',
               'Netted rock dragon', 'honey Badger', 'european Stork', 'brush-tailed Bettong', 'Black-footed ferret', 'Porphyrio porphyrio', 'Chimpanzee', 'vervet Monkey', 'Egretta thula',
               'Ring-necked pheasant', 'White-faced whistling duck', 'Paraxerus cepapi', 'otolemur', 'viper', 'Uraeginthus granatina', 'Greylag goose', 'Laniaurius atrococcineus',
               'crested Porcupine', 'pigs', 'Vicuna', 'philippinarum', 'Black-faced kangaroo', 'Halcyon smyrnesis', 'acinonyx', 'yearling', 'White-winged tern',
               'Terrapene carolina', 'Common rhea', 'Springbuck', 'European beaver', 'blue Peacock', "Coke's hartebeest", 'Sula nebouxii', 'lynx', 'carmine Bee-eater', 'crow',
               'sturnus', 'king Cormorant', 'sacred Ibis', 'duiker', 'Indian giant squirrel', 'macaws', 'donkey', 'gila Monster', 'Vervet monkey', 'elephant', 'Crested screamer',
               'common Grenadier', 'Arctogalidia trivirgata', 'southern right Whale', 'Golden jackal', 'bothrops', 'meles', 'Southern elephant seal', 'aracaris',
               'hares', 'crested', 'Alopex lagopus', 'jungle Cat', 'African fish eagle', 'Gulls', 'southern black-backed Gull', 'Balearica pavonina', 'netted rock Dragon',
               'black-tailed prairie Dog', 'mussel', 'Emu', "gambel's Quail", 'Actophilornis africanus', 'Giraffe camelopardalis', 'Taxidea taxus', 'anubis', 'Sage grouse', 'Hyaena brunnea',
               'tawny Eagle', 'whales', 'lituratus', 'Common grenadier', 'Numida meleagris', 'Hyaena hyaena', 'reduviid', 'American Virginia opossum',
               'Hippopotamus amphibius', 'owe', 'spoonbills', 'cristatus', 'Purple grenadier', 'Mudskipper', 'livia', 'manatus', 'clarkii', 'dolphin', 'Terathopius ecaudatus', 'fox', 'lesser masked Weaver',
               'Buteo galapagoensis', 'white-nosed Coatimundi', 'common Wolf', 'Savanna fox', 'black-fronted Bulbul', 'Tiger', 'puna Ibis', 'Blue duck',
               'duikers', 'Common goldeneye', 'crotalus', 'Eurasian hoopoe', 'shrew', 'agile Wallaby', 'leucoryx', 'blue Wildebeest', 'boar', 'flying foxs']

corona_names = {'corona', 'covid19', 'coronavirus', 'corona', 'coronaviruses', 'CoV',
'2019-coronaviruses', 'ncov', 'hcov-2019', 'wuhan coronavirus', 'COVID‐19', 'Severe Acute Respiratory Syndrome-Coronavirus-2', 'wuhan-2019-ncov', 'coronaviruses-2019', '2019-coronaviridaecovid', 'covid19', 'betacovs', 'coronavirus disease 2019', '2019- nCoV', 'GenBank MN908947 , 296 Wuhan-Hu-1', 'COVID- 19', 'ncovs', 'Corona Virus Disease 2019', 'ncov-2019', 'coronaviruses', '2019 nCoV', 'GenBank : MN908947.3', 'sars-cov-2', 'wuhan virus', '2019-hcov', 'coronavirus-2019', 'sars-coronavirus-2', 'COVID 2019 , SARS-CoV2', 'SARS-COV-2', '2019h-cov', 'Wuhan virus', 'sars-cov2', '2019n-cov', '2019‐nCoV', 'SARS-COV2', '2019-nCOV', 'hcov-19', 'nCoV-2019', 'SARS‐CoV‐2', '2019-ncovs', 'cov', '2019-nConV', 'sars-cov-2019', 'sars-cov-2-rbd', '2019-novel coronavirus', 'novel coronavirus', 'coronavirus disease-19', 'corona', 'covid-19', '2019-ncov', 'COVID 19', 'coronaviridae', 'Wuhan-Hu-1 coronavirus', 'SARS-Cov-19', 'coronaviral', 'Sars-CoV2', '2019ncov', 'h2019-covcov-2019', 'hcovs', 'ccov-ii', '2019nCoV', 'Wuhan-Hu-1_MN908947', 'coronavirus', 'SARS-CoV-2', 'HCoV-19', 'Wuhan Coronavirus', 'WIV1-CoV', 'COVID--19', 'Coronavirus Disease 2019', '2019-coronavirus', 'COVID-19', 'COVID‑19', 'Coronavirus Disease-2019', 'covid--19', 'HuCoV-SARS', 'coronaviridae-2019', 'Covid-19', '2019-nCov', 'Coronavirus disease 2019', 'SARS-2-CoV', 'Corona Virus Disease-2019', 'covid-2019', '2019 novel coronavirus', '2019n-CoV', 'COVID19', '2019−nCoV', 'SARS2 , GeneBank ID MN908947', 'betacoronavirus', 'Novel Coronavirus 2019', 'n2019-cov', 'nCoV -19', 'COVID-2019', '2019hcov', 'betacov', 'COVID -19', 'sars-cov-n-ntd'}

animal_diction = {}
all_animal_set = set()
plural_animal_name = ''
special_animals_dic = {'cow':'cows', 'silvestris':'silvestrium'}

for animal in animal_list:
    
    if animal in special_animals_dic.keys():
        plural_animal_name = special_animals_dic[animal]
        
    elif animal in all_animal_set:
        continue

    elif singularize(animal) == animal:
        plural_animal_name = pluralize(animal.lower())
    else:
        temp = animal.lower()
        plural_animal_name = animal.lower()
        animal = singularize(temp).lower()

    all_animal_set.add(plural_animal_name.lower())
    all_animal_set.add(animal.lower())
    animal_diction.update({animal.lower() + ":" + plural_animal_name.lower(): Animal_info()})

all_animal_set.remove('')
all_animal_set.remove('s')


corona_names = [name.lower() for name in corona_names]    
is_animale = False
number_of_articles = 50000

count_animals = 0
count_animals_per_doc = 0
cc = 0

for file in datafiles:

    with open(file,'r')as f:
        doc = json.load(f)
        
    for index, item in enumerate(doc['body_text']):

        animals_in_item = set()
        corona_names_in_item = Counter()
        
        for tok in item['text'].split(" "):

            pair = ''
            tok = tok.lower()
            is_singel = False
            if tok in all_animal_set:
                if singularize(tok) == tok:
                    plural_animal_name = pluralize(tok)
                    is_singel = True
                else:
                    temp = tok
                    plural_animal_name = tok
                    tok = singularize(temp)
                    is_singel = False
               
                if (plural_animal_name not in all_animal_set and is_singel) or (tok not in all_animal_set and not is_singel):
                    continue
                pair = tok + ":" + plural_animal_name    
                animals_in_item.add(pair)
                animal_diction[pair].overall_count += 1
                
                if (not animal_diction[pair].paragraph_info_dic) or (index not in animal_diction[pair].paragraph_info_dic.keys()):
                    animal_diction[pair].paragraph_info_dic.update({index:Paragraph_info()})
                                
                animal_diction[pair].paragraph_info_dic.get(index).count_animal += 1
                
            if tok in corona_names:
                corona_names_in_item[tok] += 1
        
        for corona_name, count in corona_names_in_item.items():
            for animal in animals_in_item:
                animal_diction[animal].paragraph_info_dic[index].count_corona_name += count
                animal_diction[animal].paragraph_info_dic[index].corona_nams_list.append(corona_name)
        
    cc += 1
    if cc == number_of_articles:
        break
        
animal_list = []
count_parag_animals = []
count_corona_related = []
overall_count_animals = []


for animal_name, animal_info in animal_diction.items():
    if animal_info.overall_count == 0:
        continue
        
    animal_list.append(animal_name.split(":")[0])
    count_parag_animals.append(len(animal_info.paragraph_info_dic))
    count = 0
    
    for _, info in animal_info.paragraph_info_dic.items():
        if info.count_corona_name > 0:
            count += 1
            
    count_corona_related.append(count)
    overall_count_animals.append(animal_info.overall_count)
    
    

def autolabel(rects,ax):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * h, '%d' % int(h),
                    ha='center', va='bottom')

def complex_bar_graph(animals_names, count_overall_animals, count_corona_related):

    N = len(animals_names[0:8])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3  # the width of the bars

    fig = plt.figure(figsize=(8, 6), edgecolor='red')
    ax = fig.add_subplot(111)

    yvals = count_overall_animals[0:8]
    rects1 = ax.bar(ind, yvals, width, edgecolor="black", linewidth=0.6, color='paleturquoise')
    zvals = count_corona_related[0:8]
    rects2 = ax.bar(ind, zvals, width, edgecolor="black", linewidth=0.6, color='cadetblue', alpha=.7)

    ax.set_ylabel('Scores')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(tuple(animals_names[0:8]), rotation=45)

    autolabel(rects1,ax)
    autolabel(rects2,ax)

    plt.show()
    

def simple_bar_graph(animals_names, all_animals):
    
    N = len(all_animals[0:8])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    fig = plt.figure(figsize=(8, 6), edgecolor='red')
    ax = fig.add_subplot(111)

    yvals = all_animals[0:8]
    rects1 = ax.bar(ind, yvals, width, edgecolor="black", linewidth=0.6, color='paleturquoise')

    ax.set_ylabel('Animal frequency')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(tuple(animals_names[0:8]), rotation=45)

    autolabel(rects1, ax)

    plt.show()
count_parag_animals1, count_corona_related1, animal_list1 = zip(*sorted(zip(count_parag_animals, count_corona_related, animal_list[:]),reverse=True))

complex_bar_graph(animal_list1, count_parag_animals1, count_corona_related1)

overall_count_animals2, animal_list2 = zip(*sorted(zip(overall_count_animals, animal_list[:]),reverse=True))

simple_bar_graph(animal_list2,overall_count_animals2)
