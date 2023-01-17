!pip install git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

from cord.tasks import Tasks

from cord.core import image, get_docs

import pandas as pd

pd.options.display.max_colwidth=300
papers = ResearchPapers.load()
Tasks.GeneticOrigin
image('../input/cord-images/covidprotease.GIF')
papers.search_2d(Tasks.GeneticOrigin[1].SeedQuestion)
papers.display('kfbrar54','nm4dx5pq', 'qed8hayx','yb6if23t', 'abqrh2aw', 'm2iiswan','th1da1bb', 'afcgqjwq', 'r5te5xob',  'dao10kx9', 'ry9wpcxo', 'xetzg7gp', 'szg12wfa', '1qkwsh6a', 'jmrg4oeb', '20zr7mtt')
papers.similar_to('nm4dx5pq')
image('/kaggle/input/cord-images/transmissiontree.png')
papers.search_2d(Tasks.GeneticOrigin[2].SeedQuestion)
papers.display('vd35a2eq', '0mobdg2p','dblrxlt1', 'hyrzder6', 'aeogp8c7', 'w67z5qof', 'cuyrw4nc', 't93xjcvm', "9t73wadp", "kz5udher", "60vrlrim", "ca6pff0p")
papers.search_2d(Tasks.GeneticOrigin[3].SeedQuestion)
papers.display('c31amc2q', 'k9rjvtcy', 'xwx9w9fi','xuczplaf','dblrxlt1', 'p1jbb1wa','w296pll9', 'mugq630z', 'tjdxn29l', 'ljllvlrd', '4ko557n1','srq1bo2v', '2inlyd0t', 'jjbez46k', 'h5sox8bq', 'lfndq85x')
papers.search_2d(Tasks.GeneticOrigin[4].SeedQuestion)
papers.display( 'k9rjvtcy', '1qkwsh6a', 'rxrlbw60', 'q8im1agz', 'ba8zx73b', 'he853mwa', '49oco16h', 'dblrxlt1', 'njundv6l')
papers.search_2d(Tasks.GeneticOrigin[5].SeedQuestion)
papers.display( 'k9rjvtcy', 'c31amc2q', '65b267ic', 'k9ygdhqg', 'xz0385np', '6h5393o9', 'brhvfsgy', '8a1cia8s', 'he853mwa')
papers.similar_to('c31amc2q')
papers.search_2d(Tasks.GeneticOrigin[6].SeedQuestion)
papers.display('ljllvlrd', 'd5l60cgc', 'b6kx9nnb', '7gs54990', 'p1jbb1wa', '971d0sir', '89fol3pq', 'oxs4o9xe', 'xuczplaf')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')