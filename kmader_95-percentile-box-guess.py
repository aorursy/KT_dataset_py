THRESHOLD_FOR_PREDICTION = 0.6
PERCENTILE_TO_KEEP = 95
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sub_df = pd.read_csv('../input/lung-opacity-classification-transfer-learning/submission.csv')
sub_df['score'] = sub_df['PredictionString'].map(lambda x: float(x[:4]) if isinstance(x, str) else 0)
sub_df.drop(['PredictionString'], axis=1, inplace=True)
sub_df['score'].plot.hist()
sub_df.sample(3)
all_bbox_df = pd.read_csv('../input/lung-opacity-overview/image_bbox_full.csv')
all_bbox_df.sample(3)
mini_df = all_bbox_df.\
             query('Target==1')[['x', 'y', 'width', 'height', 'boxes']]
sns.pairplot(mini_df,
            hue='boxes', 
             plot_kws={'alpha': 0.1})
all_bbox_df['x'].plot.hist()
right_box = all_bbox_df.query('x>450')
left_box = all_bbox_df.query('y<450')
def percentile_box(in_df, pct=95):
    return (
        np.percentile(in_df['x'], 100-pct),
        np.percentile(in_df['y'], 100-pct),
        np.percentile(in_df['width'], pct),
        np.percentile(in_df['height'], pct)
    )
right_bbox = percentile_box(right_box, PERCENTILE_TO_KEEP)
left_bbox = percentile_box(left_box, PERCENTILE_TO_KEEP)
print(right_bbox)
print(left_bbox)
fig, c_ax = plt.subplots(1, 1, figsize = (10, 10))
c_ax.set_xlim(0, 1024)
c_ax.set_ylim(0, 1024)
for i, (x, y, width, height) in enumerate([right_bbox, left_bbox]):
    c_ax.add_patch(Rectangle(xy=(x, y),
                                    width=width,
                                    height=height, 
                                     alpha = 0.5+0.25*i))
def proc_score(in_score):
    out_str = []
    if in_score>THRESHOLD_FOR_PREDICTION:
        for n_box in [left_bbox, right_bbox]:
            out_str+=['%2.2f %f %f %f %f' % (in_score, *n_box)]
    if len(out_str)==0:
        return ''
    else:
        return ' '.join(out_str)
sub_df['PredictionString'] = sub_df['score'].map(proc_score)
sub_df.sample(5)
sub_df[['patientId','PredictionString']].to_csv('submission.csv', index=False)
