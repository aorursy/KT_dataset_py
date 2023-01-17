!pip install git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from cord.core import image, get_docs

from cord.tasks import Tasks

papers = ResearchPapers.load()
papers.search_2d(Tasks.MedicalCare[1].SeedQuestion)
papers.display('djjarc0f','iwyn32tv','gzsqspse','u2iognle', 'oz3cjtql', 'seqv4xxp', 'ce1us64t',  'ajqbljai', 'kwncu3ji','k5htbxv3','yszyahj7')
papers.similar_to('kwncu3ji')
papers.display('517s290b','4kzv06e6')
papers.search_2d(Tasks.MedicalCare[2].SeedQuestion)
papers.display('y4uo7o8g', 'p0y2rpqa', '4feo08nh', 't624zyyc', 'b23301ac', 'vsinwqnr', 'wypttkea','fpp8osgs', 'v0booj4n',

               'vsinwqnr', 'zlxqt7qy', 'ilnc1n90', 'd2bh3tm8', '1ups9sk9', 'fwa6mpb9', 'modtthxx')
image('../input/cord-images/doctor_ppe_upset.jpg')
papers.search_2d(Tasks.MedicalCare[3].SeedQuestion)
papers.display('m40kwgcg', 'qdhhh9gx','0euaaspo', '2ioap802','bb1tp87n', '7styuuzo', 'fy3aet57', 'm40kwgcg', 'k36rymkv', 'vg1vvm6f', 't8azymo7', 'hsxwz798', 't996fbad', 'jir7n19b', 'e99athff')
papers.search_2d(Tasks.MedicalCare[4].SeedQuestion)
papers.display('aoviner2', '9avy4f62', 'r8bhgb7k', '8xmcljal', '2ioap802', '2opsf5f5', 'fh6f3dkn')
image('../input/cord-images/ecmo_survival.png')
papers.search_2d(Tasks.MedicalCare[5].SeedQuestion)
papers.display('moe0bi1g', 'nnlynjoy', '4mnmaky6','8prg1goh','97gawzw4','ckh8jp9e', '8udyoutb', 'skknfc6h', '2ioap802', 'h4cu63cm')
papers.search_2d(Tasks.MedicalCare[6].SeedQuestion)
papers.display('tj21zcod', 'w49i0xkz', '921orhjg','d1pgjtjh', 'ky33ju30', 'zblitbo0', 'tj21zcod', '0kss5r7u', '3egv50vb', 'jvm1pcjy', 'kn1lcsia', 'ziepfnpz', '13akn7dm', 'wztm0rbx')
papers.search_2d(Tasks.MedicalCare[7].SeedQuestion)
papers.display('88xd6pi2', 'b23301ac', '8xfy2s14', 'dham4nnk','zawfotcx','ym2s6fd4','yfrzp8ur', '6we8e60b', '88qiin0x', 'x9yo2vcs', 'kjx03hju', '2320fnd5')
papers.search_2d(Tasks.MedicalCare[8].SeedQuestion)
papers.display('3sdmqj4e', 'edspdu5x','6srm6t7d', '4yvfdqbq','ujk48pe7', 'oi290bsa', '36cs0vt9', 'ei8ad6ci', 'bpukqctg', 'dkbujz83')
papers.search_2d(Tasks.MedicalCare[9].SeedQuestion)
papers.display('vb3hygtv', '5ebfot4v', 'kyi8tfqd', '3qhpk5az', 'gil70fcj', 'vpmrwanr', 'yd29gk7q', '88xd6pi2', 'w9a0jopu', '8sd7aip4', '91weyqmm', '90zy02wu', 'j1p31gep')
papers.search_2d(Tasks.MedicalCare[10].SeedQuestion)
papers.display('011k6mm0', '2te9myos', 'exhg840i', '88xd6pi2', '6kr2uwlq', 'c9vvhe8t', 'djjarc0f', '8a1cia8s')
papers.search_2d(Tasks.MedicalCare[11].SeedQuestion)
papers.display('jwivka1t', 'fanatrfo', '15rpskir', 'sauf8p00','pgxew3yk', 'pl4e8vv9', '5fbcjhqj')
papers.search_2d(Tasks.MedicalCare[12].SeedQuestion)
papers.display('6tso9our','0him5hd2','r2s5wq5l', '2s5xd1oc','v48c2798', 'j97ugs3y', 'x70501t3', '3uk2zfpo', '3gv6469y', '96r8l6vq', 'nvavj9gk', 'azrqz6hf', 'e6q92shw')
papers.search_2d(Tasks.MedicalCare[13].SeedQuestion)
papers.display('viwochvv', 'pzjxsh02','hecqmchz', 'bxr2pkm9','2320fnd5', 'nqauuzw0', 'ce1us64t', 'slejus63', 'vsinwqnr', '0iburamm', '7jkzbmsf')
papers.search_2d(Tasks.MedicalCare[14].SeedQuestion)
papers.display('20zr7mtt', '1zyeusat', '7e8zlt3t', 'vpodtbjk', 'dbzrd23n', 't7gpi2vo', 'xuczplaf')
papers.search_2d(Tasks.MedicalCare[15].SeedQuestion)
papers.display('ufcvecwo', 'ce8hrh5e', 'yth3t2cf', 'bsddxgx2', 't1wpujpm', 'iwsa760n', '61ytvjrb', 'nrnc8u28', 'wm0forsu')
papers.display('cqlg8go7', '2z82hmfh', 'bntjg90x', '8znnq0rh', 'w3fsxg90')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')