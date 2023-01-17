!pip install -qq tensorflow_hub
import tensorflow as tf
import tensorflow_hub as hub
# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
    # We will be feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module('https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1')
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()
session = tf.Session(graph=g)
session.run(init_op)
en_embed = lambda word_vec: session.run(embedded_text, feed_dict={text_input: word_vec})
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
papers_df = pd.read_csv(os.path.join('../input/', 'uzh_papers.csv'))
papers_df['citedby-count'].hist()
papers_df.sample(3)
papers_df['timecode'] = pd.to_datetime(papers_df['prism:coverDate'])
papers_df['years_elapsed'] = (papers_df['timecode']-papers_df['timecode'].min()).dt.total_seconds()/(3600*24)/365.25
papers_df['years_elapsed'].hist()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(
    max_df=0.25, # remove shit that is everywhere
    min_df=3
) 
title_vec = papers_df['dc:title'].dropna()
cv.fit(title_vec)
word_vec = cv.transform(title_vec)
print(str(cv.vocabulary_)[:100])
word_vec
pillar_focus_dict = {
    'A) Ethics and Social Tranformation': [
        'Ethics, Social transformation, Societal changes, ethical review board',
    ],
    'B) Data Availability, Quality, and Security': [
        'Data quality control, Data protection, Data security, Data privacy'
    ],
    'C) Prediction and preemptive behavior': [
        'Predictive analysis, Preemptive behavior, Predictive modeling, Forecasting'
    ]
}
pillar_focus_vec = {k: en_embed(v) for k,v in pillar_focus_dict.items()}
title_vec = en_embed(papers_df['dc:title'].fillna('').values.tolist())
papers_df['emb_vec'] = [x for x in title_vec]
pillars = sorted(pillar_focus_vec.keys())
pillar_vec = [pillar_focus_vec[k] for k in pillars]
full_vec = np.concatenate(pillar_vec+[title_vec], 0)
full_vec.shape
from sklearn.manifold import TSNE
tsne_model = TSNE(n_iter=1500, 
                  random_state=2018, 
                  perplexity=10, 
                  learning_rate=400, verbose=True)
tsne_vec = tsne_model.fit_transform(full_vec)
ss_vec = tsne_vec[:-title_vec.shape[0]]
papers_df['tsne_x'] = tsne_vec[-title_vec.shape[0]:, 0]
papers_df['tsne_y'] = tsne_vec[-title_vec.shape[0]:, 1]
fig, ax1 = plt.subplots(1, 1, figsize = (15, 10))
i = 0
for k, c_color in zip(pillars, 'rgy'):
    for idea in pillar_focus_dict[k]:
        ax1.plot(tsne_vec[i, 0], tsne_vec[i, 1], f'{c_color}s', label=idea, ms=20, alpha = 0.5)
        i+=1

ax1.scatter(papers_df['tsne_x'], papers_df['tsne_y'], s=papers_df['citedby-count'], c='b', label='All Publications', alpha = 0.5)

ax1.legend()
ax1.axis('off')
import string
papers_df['tsne_x'] = tsne_vec[-title_vec.shape[0]:, 0]
papers_df['tsne_y'] = tsne_vec[-title_vec.shape[0]:, 1]
fig, m_axs = plt.subplots(1, len(pillars), figsize = (30, 8))
i = 0
for c_ax, k in zip(m_axs, pillars):
    for idea in pillar_focus_dict[k]:
        c_pt = tsne_vec[i]
        c_ax.plot(c_pt[0], c_pt[1], 's', label=k)
        i+=1
    papers_df['dist'] = np.square(papers_df['tsne_x']-c_pt[0])+np.square(papers_df['tsne_y']-c_pt[1])
    c_pnts_df = papers_df.sort_values(['dist']).head(100)
    c_ax.plot(c_pnts_df['tsne_x'], c_pnts_df['tsne_y'], 'b.', label='Closest Pages')
    c_pnts_df = c_pnts_df.sample(5)
    clean_titles = c_pnts_df['dc:title'].map(lambda x: x[:50])
    for (_, c_row), clean_title in zip(c_pnts_df.iterrows(), clean_titles):
        c_ax.text(c_row['tsne_x'], c_row['tsne_y'], clean_title) 
    c_ax.legend()
    c_ax.set_title(k)
papers_df.to_csv('paper_vec.csv')
i = 0
pill_list = []
for k in pillars:
    for idea in pillar_focus_dict[k]:
        pill_list += [{'tsne_x': tsne_vec[i, 0],
                  'tsne_y': tsne_vec[i, 1],
                  'term': idea,
                  'pillar': k}]
        i+=1
pd.DataFrame(pill_list).to_csv('pillars.csv')     
from matplotlib import animation, rc
rc('animation', html='jshtml', embed_limit=100)
step_count = 12
step_length = 10*1000/(step_count)
time_steps = np.linspace(papers_df['years_elapsed'].min(),
                       papers_df['years_elapsed'].max(),
                       step_count+1)
fig, ax1 = plt.subplots(1, 1, figsize = (15, 10))
pub_plot = ax1.plot(0, 0, 'b.', label='All Publications', alpha=0.75)[0]
i = 0
for k, c_color in zip(pillars, 'rgm'):
    for idea in pillar_focus_dict[k]:
        ax1.plot(tsne_vec[i, 0], tsne_vec[i, 1], f'{c_color}s', label=k, ms=10)
        i+=1
ax1.legend(loc=1)
ax1.axis('off')
x_lim = ax1.get_xlim()
y_lim = ax1.get_ylim()
def update_frame(i):
    n_rows = papers_df[papers_df['years_elapsed']<=time_steps[i+1]]
    n_rows = n_rows[n_rows['years_elapsed']>time_steps[i]]
    #ax1.plot(n_rows['tsne_x'], n_rows['tsne_y'], 'b.', alpha = 0.25)
    pub_plot.set_xdata(n_rows['tsne_x'])
    pub_plot.set_ydata(n_rows['tsne_y'])
    ax1.set_title(n_rows['prism:coverDate'].iloc[0])
    #ax1.set_xlim(*x_lim)
    #ax1.set_ylim(*y_lim)
ani = animation.FuncAnimation(fig, 
                              update_frame, 
                              range(step_count), 
                              interval=step_length)
ani
