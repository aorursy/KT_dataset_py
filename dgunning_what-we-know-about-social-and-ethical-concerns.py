from ipywidgets import Image

f = open("../input/cord-images/medicalethics.jpg", "rb")

Image(value=f.read())
!pip install -U git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from IPython.display import display

from cord.core import image, get_docs

from cord.tasks import Tasks

papers = ResearchPapers.load()
papers.search_2d(Tasks.EthicsSocial[1].SeedQuestion)
papers.display("sxc6uue6", "crtg1heh", "meq5b1z1", "vahd59z8", "6vf3km5i", "kdf654z4",  "58czem0j", "sg6vnur0",

                        "cmw0lgpd", "vb6z0hzb",  "l0fyxtva", "6vf3km5i", 'fmg6hdm0',

                        "pqr5keuu", "woscjkii",  "io7ozfpi", "8fwa2c24")
papers.search_2d(Tasks.EthicsSocial[2].SeedQuestion)
papers.display("reo51jgp", 'pc4l0vf2', 'v0fdlt82', '2320fnd5', '22tfpjm5', 'y90w53xo', '6we8e60b', '4x0g8tbz', '6rmfk63q', '0yzrcfe5')
papers.search_2d(Tasks.EthicsSocial[3].SeedQuestion)
papers.display('2320fnd5','elbjs7ft', 'meq5b1z1', 'h0oqnue8', 'woscjkii',  'rftvrs8w', 'bcyotwkf', 'cy87puhu', 'zgdxjwes', 'heplijn8')
papers.search_2d(Tasks.EthicsSocial[4].SeedQuestion)
image('../input/cord-images/who_collaborating_centres.png')
papers.display('nwt6snyr', '9hnm8u22', 'igf0gpu5', '2bo0s0hz', 'mqbud21t', '3fflt6a3')
papers.similar_to('nwt6snyr')
papers.display("t3rw6sv9", "7z10l0m1", "h9o2d9jv")
papers.search_2d(Tasks.EthicsSocial[5].SeedQuestion)
papers.display('1152vpv5', 'ncot7tvn', 'e4uyh2zb', 'bs4jczmw', 'si9jhugj', 'ass2u6y8','7yucn30u',

                        'jg608vms',  '6e0xfi8b', 'grmef7ab', '28utunid', 'urkuofmu', 'jab7vp33', '09ym98hx')
papers.search_2d(Tasks.EthicsSocial[6].SeedQuestion)
papers.display('uok3n79f','nwtfs1t9','o72unm3q', "s73hljz9", '5k5whv09',"x1gces56", "465ltmml", 'yzixu404', '5k5whv09','lej3v5lb', 'cfmgoy14', 'r8tlzlal')
papers.similar_to("s73hljz9")
papers.search_2d(Tasks.EthicsSocial[7].SeedQuestion)
papers.display("352gisfw", "kxd03q02", "m192lw2f",'72id80ab','09ym98hx', "95zawygq", 'vsn43pxu', '5h3bbtzs','0lyxvex0','rgiuipyj',

                        "cout9w7r", "ik2wquyk", "c9xs7idm", "h0boxuse", 'c8nrd4h9', '1jsnb485', 'vsn43pxu', 'xwcaycmq', '16ciqu9w', '7yucn30u')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')