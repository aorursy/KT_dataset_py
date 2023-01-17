# content of article from https://chicago.suntimes.com/business/2020/1/28/21112412/uic-grant-melinda-gates-break-through-tech-women-computer-science



article = '''A program funded by Melinda Gates is investing in women in tech — and it’s starting in Chicago.



Cornell Tech’s Break Through Tech will partner with the University of Illinois at Chicago to “develop educational opportunities for women and other underrepresented groups at UIC,” the university said.



The university is the first to be selected for the program’s national expansion and received one of the “largest grants the UIC College of Engineering has ever received” to implement it, the university said in a news release.



The university did not disclose the grant amount.



The program will help freshmen and sophomore computer science students secure paid internships by partnering with local companies and industry players, which in turn, should help them get jobs.



“We want to see a 12.5% increase in the representation of women graduating from our undergraduate computer science program and we want 100 percent of those women to have meaningful internships on their resumes when they graduate,” Robert Sloan, UIC professor and head of computer science, said in the release.



The internship programs will be held during the university’s winter breaks, dubbed ‘Winternships.’



Introductory courses and workshops will begin in May, and the first batch of students participating in the internship program could start as soon as December.



The program works with Pivotal Ventures, a Melinda Gates investment company, and started a similar initiative with City University of New York called Women in Technology and Entrepreneurship in New York.



UIC was chosen as a partner of interest by Pivotal Ventures due to the scale and growth of its computer science program and the student body’s overall diversity, university officials said.'''
import gensim

from IPython.display import display, HTML

import pandas as pd
def summarize_text(text, n_sentences=5):

    '''Produce a summary of a text by extracting key sentences

    

    Parameters

    ----------

    text : str

        The text to summarize

    n_sentences : int, optional

        The number of sentences in the summary. Default is 5.

        If the original text is shorter than n_sentences, the original text will be returned.

    

    Returns

    -------

    summary : str

    '''

    n_sentences_original = len(list(gensim.summarization.textcleaner.get_sentences(text)))

    if n_sentences_original <= n_sentences:

        return text

    summary = gensim.summarization.summarize(text, ratio=n_sentences/n_sentences_original)

    return summary
summary = summarize_text(article)

display(HTML(summary))
def compare_article_and_summary(article, summary):

    '''Print a side-by-side comparison of the original article and its summary

    

    Parameters

    ----------

    article : str

    summary : str

    '''

    article_sentences = list(gensim.summarization.textcleaner.get_sentences(article))

    summary_sentences = set(gensim.summarization.textcleaner.get_sentences(summary))

    side_by_side = pd.DataFrame({'article': article_sentences, 'summary': [s if s in summary_sentences else '' for s in article_sentences]})

    display(HTML(side_by_side.to_html()))
compare_article_and_summary(article, summary)