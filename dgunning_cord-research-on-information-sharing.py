from ipywidgets import Image

f = open("../input/cord-images/medical_information_sharing.jpg", "rb")

Image(value=f.read())
!pip install -U git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from IPython.display import display

from cord.core import image, get_docs

from cord.tasks import Tasks

research_papers = ResearchPapers.load()
research_papers.searchbar()
research_papers.search_2d(Tasks.InformationSharing[1].SeedQuestion)
research_papers.display('4nmc356g', 'zp4oddrt', 'j59tm40d')
research_papers.similar_to('4nmc356g')
research_papers.display('szg12wfa', '8m06zdho', '7vuejy26', 'cbc98t7x', 'ry9wpcxo', 'tnw82hbm')
research_papers.display('y691hark', 'rjc3b4br',  'r0lduvs1', '7i422cht', 'pa9h6d0a', 'dbzrd23n', '5gbkrs73', '94tdt2rv', 

                        'xsgxd5sy', 'jf36as70', 'uz91cd6h', 'cd9lf27c', '6xkm2j0f', '9sk11214', 'qeehgxa1')
research_papers.search_2d(Tasks.InformationSharing[2].SeedQuestion)
research_papers.display("kz5udher", 'fghu8ouc', 'c5tw8458', 'zjsjgd31', '3fflt6a3', '94tdt2rv', 'dao10kx9', 'mnhoyz35', 'jxwpagrb', 'k0pzlnt9',

                        'li023qda', 'cd9lf27c', '0z8x6v04', 'nowwviz3', 'jwqrfb6h')
research_papers.search_2d(Tasks.InformationSharing[3].SeedQuestion)
research_papers.display('mqbud21t', 'kz5udher','cd9lf27c', '637rsa3g','zp4oddrt','2r4uty9g','4y5279c5', 'vkn60zzs', '15drpph0', '5oc0lisi', '94tdt2rv', 'j58f1lwa', 'skrack0x')
research_papers.search_2d(Tasks.InformationSharing[4].SeedQuestion)
research_papers.display('c5tw8458','elbjs7ft', 'nxyv3gun', 'i0gp9ehw', 'n6ptr4wz', 'kjd2huab', 'c4l7qhh4', '1r184p3g', '4so0jlrx', 'kw0y9fpp')
research_papers.display('g8lsrojl')
research_papers.search_2d(Tasks.InformationSharing[5].SeedQuestion)
research_papers.display('nxyv3gun', 'catxtvrm', 'mh9bk9z2', 'ticf2ilm', 'u20p22ch', 'c5tw8458', 'nu8q72l9', 'lss2igfw', '37ejmq3l',

                        'l4o7nicc', 'vj2z52vg', '9h4pq7up', 'y691hark', 'lb0fd7ig')
research_papers.similar_to('nxyv3gun')
research_papers.search_2d(Tasks.InformationSharing[6].SeedQuestion)
research_papers.display('naavpb4v','s8fvqcel', 'gwhb2e6q', '5x59kfb3', 'zsy5gowg', 'gn0xm2gy', 'h4yivj9l', '3fflt6a3', 'ptgpxpad', '4so0jlrx')
research_papers.search_2d(Tasks.InformationSharing[7].SeedQuestion)
research_papers.display('76cygq88','c2lqffbf', 'kwncu3ji', 'ce1us64t', '0p383h4s', '91lzomwp', '9bs1udnv', 'ejgr63n4','lkbwxo41', 'z5jnd0bk',

                        '98jz8tox', 's73hljz9', '6r27qzap', 'kwncu3ji', 'u2iognle', 'r8tlzlal', 'n73o8i0w', 'o81b9htu', 'vv3ipuom')
research_papers.display('folvd24f')
research_papers.display('91lzomwp')
from cord.tasks import Tasks

import pandas as pd

pd.options.display.max_colwidth=200
research_papers.search(Tasks.InformationSharing[7].SeedQuestion, view='table')
research_papers.search_2d(Tasks.InformationSharing[8].SeedQuestion)
research_papers.display('ew8lwmsn','kweh1doo','dl45plby', 'gj10u4lf', 'dhdyxnr3', 'g1zynbul',  'q6og879d', 'rtybezro',  '91weyqmm', '8d1ob1mi')
research_papers.similar_to('ew8lwmsn')
research_papers.search_2d(Tasks.InformationSharing[9].SeedQuestion)
research_papers.display("plfjkp5f", 'w49i0xkz', 'ivwn4nhl', 'gttuxtw6', 'ip9geldg', 'vg1vvm6f',  '3egv50vb', 

                        'arhqix9h', 'skknfc6h', '7j4z8ye3', 'j58f1lwa', '9yrxd6bi', 'zndtddty', 'xsgxd5sy', 'ysbopqqq', 'czuq8rw5')
research_papers.search_2d(Tasks.InformationSharing[10].SeedQuestion)
research_papers.display('g1zynbul', 's96xufes', '07l9hqsr', 'lb0fd7ig', 'dl45plby', 'raelqb6j', 'tqeyx7yn')
research_papers.search_2d(Tasks.InformationSharing[11].SeedQuestion)
research_papers.display('7t5mm8og', 'a2srympy', '3fflt6a3', 'elbjs7ft', '2320fnd5', 'emodr41j', 'nowwviz3')
image('../input/cord-images/cdc_action_plan.png')
research_papers.similar_to('7t5mm8og')
research_papers.search_2d(Tasks.InformationSharing[12].SeedQuestion)
research_papers.display('350d3be2', '00kzst6d', 'qww6pe61', 'vocsxblm', 'fjlbescr', 'h3mei542', 'rj9oj9ky', 'bf098qcr', '9keet8ih', 'ptizke03')
research_papers.search_2d(Tasks.InformationSharing[13].SeedQuestion)
research_papers.display("oribgtyl", '3qqzthx8', '593wdhw5', 'voi3i5w7', '593wdhw5', '5lauop7l')
research_papers.search_2d(Tasks.InformationSharing[14].SeedQuestion)
research_papers.display( 'emodr41j', 'gwxu6dz2', 'oribgtyl', 'xnce010o')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')