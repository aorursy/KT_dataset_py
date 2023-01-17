from ipywidgets import Image

f = open("../input/cord-images/riskfactors.png", "rb")

image = f.read()

Image(value=image)
!pip install -U git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from IPython.display import display

from cord.core import image, get_docs

from cord.tasks import Tasks

papers = ResearchPapers.load()
papers.search_2d(Tasks.RiskFactors[1].SeedQuestion)
papers.display('a9lw6vrs', 'wzwa3f8f', 'gpql06um', 'be7hez61','gah4krwf', 'yx8b2moc','oqo8pa1p','std4jddn','4q2olzx1', '566pbuic', 'jg6v644y', 'w3fsxg90', '4mnmaky6','nrdiqees', 'wlat2rrl', 'qlkt5fzp', 'k36rymkv')
papers.display( '8g70j0qw', 'ei2b8oqn', 'q5x3bask', '0xciml6s', '2q6qmex3')
papers.similar_to('2q6qmex3')
papers.display('jslfu3qt', 'r5jz7966', 'x23ej29m', 'yqc43a5t', 'be7hez61', 'wk4zxmz9', '6wolrfvk')
papers.similar_to('r5jz7966')
papers.display('ktc6sp3f', '1ntplgl6', '5h6mluf0', '28ddm9um', 'ub7x9j3d', 'ttbhg0gp', 'j0nm444m', 'd4dnz2lk', 'j7hih6o9')
papers.similar_to('1ntplgl6')
papers.search_2d(Tasks.RiskFactors[2].SeedQuestion)
papers.display('e0wzkhbz','85chebix', '8t55lq43','9mact9br', 'zc7oxui5','kjl04qy5','2ui36lvl', '66ulqu11','st5vs6gq','n11fcwmw', '96wkqutc', 'zz4cczuj', 'e9rt7z8d', '36fbcobw', 'ragcpbl6', '0hnh4n9e', '56zhxd6e', 'cq6mivr9', 'ur7tu7g1', 'v3gww4iv',  'xdjgjeb9')
papers.search_2d(Tasks.RiskFactors[3].SeedQuestion)
papers.display('dop8knqn','ent5oimq','yw7psy0a', '97gawzw4', 'dcf6bl8f','od8s0zhm','xyowl659',  'oeqf7uk9','g0vyzsga',  'hypyxzk2', 'aoi4iqkf', 'ahxnhutv','plfjkp5f', '6fbnyntq', 'b4kzgubs', 'ohba2n2o', 'wxm8kqnd', '5gbjdr0u', '8q1veo3q', 'ip9geldg')
papers.search_2d(Tasks.RiskFactors[4].SeedQuestion)
papers.display('mjhez5im','gzy4yud3','jtszdioh', 'uqq27wyq','0z8x6v04','bbn1tfq5', 'tucolje2', '8f76vhyz', 'gm1mb8w5', 'uei9zw6q', '9fd5a49o', 'gv8wlo06', '9af3fk0i', '38f8ftmh')
papers.similar_to('gv8wlo06')
papers.search_2d(Tasks.RiskFactors[5].SeedQuestion)
papers.display('gc5ieskk','nn1rgdw0','09ym98hx','ufeiv47z','k9cg0efh', '6r6zwfoy', 'urkuofmu','b8a78ym6','a6ldr0mn', '4fkb1udl', 'e7hefkd3', 'e85xdrcw', '2rq32rsq', 'hgn6o5yf', '4so0jlrx', 'fqu54j4h', 'yrq58n4k')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')