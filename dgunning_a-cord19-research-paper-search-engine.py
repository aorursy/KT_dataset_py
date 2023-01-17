!pip install -U git+https://github.com/dgunning/cord19.git 
from cord import ResearchPapers
research_papers = ResearchPapers.load()
research_papers.searchbar("mother to child transmission")
research_papers.searchbar("mother to child transmission", view='table')
research_papers.search('vaccine effectiveness')
research_papers.query('journal =="New Scientist" & published > "2020-02-29" ') 
research_papers.head(4)
research_papers.tail(4)
research_papers.since_sars()
research_papers = ResearchPapers.load(index='texts')
research_papers.searchbar('disease modelling pandemics')
research_papers['8m06zdho']
from cord.vectors import similar_papers
similar_papers('8m06zdho')
query = """

Since its emergence and detection in Wuhan, China in late 2019, the novel coronavirus SARS-CoV-2 has spread to nearly

every country around the world, resulting in hundreds of thousands of infections to date. To uncover the sources of SARS-CoV-2 

introductions and patterns of spread within the U.S., we sequenced nine viral genomes from early 

reported COVID-19 patients in Connecticut. By coupling our genomic data with domestic and international travel patterns, 

we show that early SARS-CoV-2 transmission in Connecticut was likely driven by domestic introductions. Moreover, the risk of domestic 

importation to Connecticut exceeded that of international importation by mid-March regardless of our estimated impacts of federal travel restrictions. This study provides evidence for widespread, sustained transmission of SARS-CoV-2 within the U.S. and highlights the critical need for local surveillance.

"""

research_papers.search_2d(query)