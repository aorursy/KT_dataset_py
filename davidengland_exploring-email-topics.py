import pandas as pd
import sqlite3
import re
from gensim.summarization import summarize, keywords

con = sqlite3.connect('../input/database.sqlite')
e = pd.read_sql_query("Select ExtractedBodyText From Emails where ExtractedBodyText", con)
status_re = re.compile("^\d{1,2}[:|.]\d{1,2} [a|p]m .*")
non_alpha_re = re.compile('[^a-zA-Z .]')
texts = filter(lambda s: not status_re.match(s), e.ExtractedBodyText)
texts = [non_alpha_re.sub('', text) for text in texts]
texts = filter(lambda x: x, texts)
texts = filter(lambda text: len(text) > 100, texts)
for text in texts:
    print(text)
    print("Summary:\n{}".format(summarize(text, ratio=0.3)))