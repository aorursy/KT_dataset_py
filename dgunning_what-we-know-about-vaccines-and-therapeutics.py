from ipywidgets import Image

f = open("../input/cord-images/coronavirus_vaccine.png", "rb")

Image(value=f.read())
!pip install git+https://github.com/dgunning/cord19.git
from cord.tasks import Tasks

from cord.core import image, get_docs

from cord import ResearchPapers

import pandas as pd

pd.options.display.max_colwidth = 100
papers = ResearchPapers.load()
papers.search_2d(Tasks.Vaccines[1].SeedQuestion)
papers.display('4zmqplz0', '6dq8xx7c', 'fnrm6a79','aosmo568','rl3801n6', 'sjyrr2bn', 'hq5um68k', 'pidar1gz', 'tyx7yuek', 'hq5um68k','5f95gve3', '68sbqdi3', 'bbumotlt', 

               'nbjili3v', 'b518n9dx',  'n6l2804j', 'a5udnv5f', 'qebbkr6d')
papers.similar_to("pidar1gz")
papers.search(Tasks.Vaccines[1].SeedQuestion + ' Chloroquine and hydroxychloroquine', covid_related=True)
papers['i8lgih81']
papers['098kmfms']
papers.search_2d(Tasks.Vaccines[2].SeedQuestion)
papers.display('d213qdsy', '405jvqyv', 'za3qypgg', 'ju35nyir',  '9igk3ke1', '1vlowf2n', '1vm5r7pq', 'zb434ve3',

              'pidar1gz', 'hj675z1b', 'uyu2buo1', 'qefiw4ho', 'nn15iyqd', 'hq5um68k', '6zk0ioep', '29h189tx')
papers.search(Tasks.Vaccines[2].SeedQuestion + ' drug repositioning repurposing', covid_related=True)
papers.search_2d(Tasks.Vaccines[3].SeedQuestion)
papers.display('ctd9sutv', 'yinweejd')
papers.display('ctd9sutv')
papers.display('mx30g5w8', 'a4qqfguo', 'q4zuslmp', 'fnguelau', 'yzffm05r', '8o884nyp')
papers.search_2d(Tasks.Vaccines[4].SeedQuestion)
papers.display('68sbqdi3', 'mfiaubqb', 'ptnmtvzj', 'm5ho8jqp', 'sdiwnhs0', 'e4x0ss66', '1fgnfh62')
papers.similar_to('68sbqdi3')
papers.search_2d(Tasks.Vaccines[5].SeedQuestion)
papers.display('nbjili3v', '9ryu9ady', 'uq2gvye9', 'fjmchbew','hf3paocz', 't1wpujpm', '9igk3ke1', 'ptnmtvzj', '95fc828i', 'hq5um68k', 'za3qypgg')
papers.search_2d(Tasks.Vaccines[6].SeedQuestion)
papers.display('95r8swye', 'mrgw2mnx', 'odcteqg8', 't8bobwzg', '75yqvjdk', '80e9anz1', '4nchg95h', 'mfiaubqb','e4x0ss66',

               'aj1cod3x', 'uxl8liil', 'kxjeallw')
papers.search_2d(Tasks.Vaccines[7].SeedQuestion)
papers.display('xnphkn3s', 'kea2xw1g', 'uyoerxvu', 'd3bs7yzc', '68sbqdi3', 'lmstdmyb', '8gncbgot', '13jupb26', 'cm30gyd8')
papers.similar_to('13jupb26')
papers['by4aefjc']
papers.display(papers.contains('Heeney', 'authors').metadata.cord_uid.tolist())
papers.search_2d(Tasks.Vaccines[8].SeedQuestion)
papers.display('68sbqdi3', 'm5ho8jqp', '6ojmmmuj', 'hq4jb2wy', 'zn182czp', 'fnrm6a79', 'sdiwnhs0', 'lzdwpaki')
papers.search_2d(Tasks.Vaccines[9].SeedQuestion)
papers.display('kwf6a2x1', 'w49i0xkz', 'drj3al9t', 'tun1ndt4', 'c5f5prkn','elbjs7ft', 'lxkxgbun','2320fnd5', '58czem0j', 'ltp7iv1z', 'yy7abob9')
papers.similar_to('kwf6a2x1')
papers.search_2d(Tasks.Vaccines[10].SeedQuestion)
papers.display('yb5kf0u2', 'm5ho8jqp', 'zn182czp')
papers.similar_to('yb5kf0u2')
papers.search_2d(Tasks.Vaccines[11].SeedQuestion)
papers.display('68sbqdi3', 't4v368v2', '2uwnamao', '7i52vltp', 'eifrg2fe', '2uvibr2j', '6ojmmmuj', 'vlqd192g', 'xjr93dm6', 'e8qubwha', 'ld0vo1rl')
papers.similar_to('vlqd192g')
papers.display('2uvibr2j', '80m78jh2', 'b3ui95vx')
papers['y3ia8g3h']
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')