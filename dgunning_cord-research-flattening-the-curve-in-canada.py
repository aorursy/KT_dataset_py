!pip install git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers

papers = ResearchPapers.load(index='text')
papers
covid_papers = papers.since_sarscov2()
covid_papers.searchbar('relationships between testing tracing efforts and public health outcomes')
papers.match('.*Romer, ?P', column='authors')
papers[32213]