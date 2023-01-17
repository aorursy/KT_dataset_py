from ipywidgets import Image

f = open("../input/cord-images/covidtesting.png", "rb")

Image(value=f.read())
!pip install -U git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from IPython.display import display

from cord.core import image, get_docs

papers = ResearchPapers.load()
from cord.tasks import Tasks
papers.search_2d(Tasks.DiagnosticsSurveillance[1].SeedQuestion)
papers.display('6vt60348', 'kfbrar54',  'p7a8chn4',  'zeq0h02t', 'noejyvtd', 'ivwn4nhl', '5b2l7xxc',

                       'htskkuox', 'hq8dg87u', 'hw2ktcsh', 'th1da1bb')
papers.search_2d(Tasks.DiagnosticsSurveillance[2].SeedQuestion)
papers.display('tf34z0x6','bx2xspbe', '6l1fyl93', 'z6cridqh', 'agzb8aac','m0xiq2aj', 'hzaf52l4', 'kduisp14', '2int7x6k', '91uj6sph','mcyeyl4s', 'vjiai5sx')
papers.display('su7f5ges', '5gbkrs73', '5j4nt2qs', '9595vm0k', '6l1fyl93', 'e6q92shw', 't8s4s0wo', 'dycx8i7h', '31i1k332')
papers.display('0syudviv')
papers.search_2d(Tasks.DiagnosticsSurveillance[3].SeedQuestion)

papers.display('z6pwfshq','kw0y9fpp','2320fnd5', 'c429kxqr', '4so0jlrx', 'tcecnc54', 'er6q40d3', 'nrnc8u28', 'uql62u7l', 'z6pwfshq')
papers.similar_to('z6pwfshq')
papers.search_2d(Tasks.DiagnosticsSurveillance[4].SeedQuestion)

papers.display('l4o7nicc', 'xn59h5d0', 'vj2z52vg', 'b6219mp3', '1mzq4llc', 'y691hark', 'pzjxsh02')
papers.search_2d(Tasks.DiagnosticsSurveillance[5].SeedQuestion)

papers.display('oc0edgd2','fnlgeubu', 'modtthxx', 'mskkahwi', '63ddlh20', 'ybbnq323','oc584x1k','a9clzlzb','8gncbgot', 'pg6893i5', '6kpgt70s', 'b8eezxv3', '5xki1ulf')
papers.search_2d(Tasks.DiagnosticsSurveillance[6].SeedQuestion)

papers.display('a0inkl9e', '50dbw0o5', 'zlhu1jit', 'rkkhv4mw', 'qu9b07ea')
papers.display("o4z6xthw",'8cg5yj20', 'blax9sz2', 'mrgw2mnx', 'wuvry51z', 'n9i54wzh', 'ec3egn8o', 'eui41zyg', 'ezv6xp16')
papers.similar_to("zlhu1jit")
papers.search_2d(Tasks.DiagnosticsSurveillance[8].SeedQuestion)
papers.display('63bos83o','gfwqog3x','qeehgxa1', '0nh58odf', 'a4mf2vnp', 'wubaahn2', 's6hld5xv', '0q928h3b', 'kzn8lwzu', 'cv3qgno3')
papers.search_2d(Tasks.DiagnosticsSurveillance[9].SeedQuestion)
papers.display('mskkahwi', 'z4l3pk23', 'scc9wee0','b8eezxv3', '09r4d3nu', 'aa7slcnc', '3c4tncj5', '0gkoanrg', 'gunn55f9', '50pupre2', 'kb881liy')
papers.similar_to("emrwcx0m")
papers.search_2d(Tasks.DiagnosticsSurveillance[10].SeedQuestion)
papers.display('c3dxfet9', '6gwnhkn4', 'x5btuxrq','0pigqtzt', 'tv9xsned',  '50oy9qqy', 'w3fsxg90',  '68193u0a', 'kbd2h4l0')
papers.similar_to('c3dxfet9')
papers.similar_to('50oy9qqy')
papers.similar_to('nifz133q')
papers.search_2d(Tasks.DiagnosticsSurveillance[11].SeedQuestion)
papers.display('4gaa14ly', 'gkd1h8yi', 'bgm3bt78', 'jsbdmnx5', 'oee19duz', 'x22rc60j', '6l1fyl93', 'rpkyycru')
papers.search_2d(Tasks.DiagnosticsSurveillance[12].SeedQuestion)
papers.display('jpb9a8c4', 'su7f5ges', 'vou46eie', 'si1yyg5e', 'yj9tfemw', 'mi6w8ppe','modtthxx')
papers['su7f5ges']
papers.similar_to('jpb9a8c4')
papers.search_2d(Tasks.DiagnosticsSurveillance[13].SeedQuestion)
papers.display('revhbd0q', "3begdfx2", "5k3hq2e7", "6arjb38j", 'm0q7rm6z', '3wuh6k6g','fnlgeubu', 'ym73y41t', 'ym73y41t')
papers.search_2d(Tasks.DiagnosticsSurveillance[14].SeedQuestion)
papers.display('uijwq8gx','20hk99h4', '2hbcbvt6', 'eui41zyg', '58ta20fg', 'fnlgeubu', 'ntgt5hy9', 'avo5jfaj', 'jsbdmnx5', 'lxjhz079', '0tetqt33', 'er6q40d3', 's4kfza3o')
papers.similar_to('lxjhz079')
papers.search_2d(Tasks.DiagnosticsSurveillance[15].SeedQuestion)
papers.display('si1yyg5e', 'ne73ykbk','qse41ybm', '2una9767','yb6if23t', 'xgwa5uoy','3sr4djft','vlqd192g', 'xieqswct', 'hth8f5sn', 'gzubfnqg', 'm9d90tgq', '6kpgt70s', '0l33i6s4', 'n9i54wzh')
papers.search_2d(Tasks.DiagnosticsSurveillance[16].SeedQuestion)
papers.display('s4kfza3o', 'q6h1f69g','modtthxx', '6ji8dkkz', '8kccpd4x', 'u2uswgy3', '68ps3uit', 'abqrh2aw', 'icpwfyss', 'o47us1cx', 'q6w8zgc3')
papers['68ps3uit']
papers.similar_to('68ps3uit')
papers.search_2d(Tasks.DiagnosticsSurveillance[17].SeedQuestion)
papers.display('fu3cl7lq', '0hxan9rw', 'izkz1hz4', 'txuthqgo', 'c2lljdi7','3r8jbhhq', '0hxan9rw', 'eifrg2fe')
papers.search_2d(Tasks.DiagnosticsSurveillance[18].SeedQuestion)
papers.display('fway438n', '41m5b96p', 'z01saucu', 'jf36as70', 'qeehgxa1', 'd9v5xtx7')
papers.search_2d(Tasks.DiagnosticsSurveillance[19].SeedQuestion)
papers.display('2inlyd0t', 'jkm496ip', '4ihv80au', 'yaspd6l7', '5f42du0b', '1qkwsh6a', 'he853mwa', 'bnuda70x', 'hidirfkv')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')