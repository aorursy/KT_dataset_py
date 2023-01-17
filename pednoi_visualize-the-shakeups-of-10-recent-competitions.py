import numpy as np

from scipy.stats import skew, kurtosis



import seaborn as sns

import matplotlib.pyplot as plt



import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from html.parser import HTMLParser
class RankParser(HTMLParser):



    def __init__(self):

        self.entered = False

        self.sign = ''

        self.data = []

        

        super().__init__()

    

    def handle_starttag(self, tag, attrs):

        if tag == 'td' and ('data-th', 'Change') in attrs:

            self.entered = True

            

        if self.entered and tag == 'span':

            if len(attrs) > 0 and len(attrs[0]) > 1 and attrs[0][1].startswith('position-change'):

                direction = attrs[0][1][len('position-change__'):]

                if direction == 'fallen':

                    self.sign = '-'                    



    def handle_endtag(self, tag):

        if self.entered and tag == 'td':

            self.entered = False

            self.sign = ''



    def handle_data(self, data):

        if self.entered:

            data = '0' if data == '—' else data

            self.data.append(int(self.sign+data.strip()))

    

    def get_data(self):

        return self.data



def read_html(file_path):

    content = open(file_path, encoding='utf-8').read()    

    parser = RankParser()

    parser.feed(content)

    return parser.get_data()
files = [('santander', 'Santander Customer Transaction Prediction _ Kaggle.html'),

         ('vsb', 'VSB Power Line Fault Detection _ Kaggle.html'),

         ('malware', 'Microsoft Malware Prediction _ Kaggle.html'),

         ('whale', 'Humpback Whale Identification _ Kaggle.html'),

         ('elo', 'Elo Merchant Category Recommendation _ Kaggle.html'),

         ('quora', 'Quora Insincere Questions Classification _ Kaggle.html'),

         ('protein', 'Human Protein Atlas Image Classification _ Kaggle.html'),

         ('plasticc', 'PLAsTiCC Astronomical Classification _ Kaggle.html'),

         ('quickdraw', 'Quick Draw Doodle Recognition Challenge _ Kaggle.html'),

         ('salt', 'TGS Salt Identification Challenge _ Kaggle.html'),]



files = [(f[0], f'../input/{f[1]}') for f in files]



rank_diff = []

          

for f in files:          

    rank_diff.append((f[0], read_html(f[1])))
def plot_hist(title, diff):

    stats = ""

    stats += "count = %d\n" % len(diff)

    stats += "mean = %.2f\n" % np.mean(diff) # always zero because the data are zero-sum

    stats += "std = %.4f\n" % np.std(diff)

    stats += "skew = %.4f\n" % skew(diff)

    stats += "kurtosis = %.4f\n" % kurtosis(diff)

    

    fig = plt.figure(figsize=(16, 4))

    sns.distplot(diff, bins=100)

    plt.text(0.05, 0.5, stats, transform=plt.gca().transAxes)

    plt.title(title)

    plt.show()
for rd in rank_diff:

    plot_hist(rd[0], rd[1])
fig = plt.figure(figsize=(15, 6))  

sns.barplot([rd[0] for rd in rank_diff], [np.std(rd[1]) for rd in rank_diff])

#plt.xticks(rotation=90)

plt.show()
def plot_candle(title, diff):

    closes = np.array(range(len(diff)))+1

    opens = closes + np.array(diff)

    highs = np.where(np.array(diff)<0, closes, opens)

    lows =  np.where(np.array(diff)>=0, closes, opens)

    

    hovertext = ['private lb: '+str(c)+'<br>public lb: '+str(o) for o, c in zip(opens, closes)]



    trace = go.Ohlc(x=list(range(1, len(diff)+1)), open=opens, high=highs, low=lows, close=closes,

                    increasing=dict(line=dict(color='#FF6699')), decreasing=dict(line=dict(color='#66DD99')),

                    text=hovertext, hoverinfo='text')

    

    layout = go.Layout(

        title = "<b>%s</b>" % title,

        xaxis = dict(

            title='final ranks',

            rangeslider = dict(visible=False)

        ), 

        yaxis=dict(

            title='shakeups',

            autorange='reversed'

        )

    )

    

    fig = go.Figure(data=[trace], layout=layout)    

    iplot(fig, filename='shakeup_candlestick')
for rd in rank_diff:

    plot_candle(rd[0], rd[1])