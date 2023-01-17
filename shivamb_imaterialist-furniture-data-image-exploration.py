from IPython.core.display import HTML 
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
# read the dataset 
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
val = pd.read_json('../input/validation.json')

# function to format the dataset 
def format_dataset(df):
    df['image_id'] = df.annotations.map(lambda x: x['image_id'])
    df['label_id'] = df.annotations.map(lambda x: x['label_id'])
    df['url'] = df.images.map(lambda x: x['url'][0])
    df.drop(columns=['annotations', 'images'], inplace=True)

format_dataset(train)
format_dataset(val)

# view the dataset snapshot
train.head(10)
plt.figure(figsize = (15, 8))
plt.title('Distribuition of different labels in the train dataset')
sns.distplot(train['label_id'], color="red", kde=False);
label_df = train.label_id.value_counts().reset_index()
label_df['index'] = label_df['index'].astype(str)
plt.figure(figsize=(15,8));
sns.barplot(x=label_df['index'][:20], y=label_df['label_id'][:20], palette="Reds_d");
# function to create images 
def display_urls(url_list, label, vals):
    img_style = "width: 180px; height:180px; margin: 0px; float: left; border: 1px solid #222;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in url_list.iteritems()])
    html = "<h3>Images related to Label: " + str(label) + " (Total Images: " + str(vals) + ")</h3><br>" + images_list + "<br><br><br>"
    display(HTML(html))

val_list = list(label_df['label_id'])
for i, label in enumerate(label_df['index']):    
    url_list = train[train['label_id'] == int(label)].url.head(4)
    display_urls(url_list, label, val_list[i])
    if i == 10:
        break
results = {'12': {'url': u'https://img11.360buyimg.com/imgzone/jfs/t5119/73/1946159194/182429/9e2c2f9e/591674d9N548522ca.jpg', 'index': [u'table lamp', u'lamp', u'furniture', u'source of illumination', u'light brown color'], 'values': [0.909, 0.957, 0.957, 0.799, 0.846]}, '20': {'url': u'https://img14.360buyimg.com/imgzone/jfs/t2320/137/487468866/439466/bb0f64ce/561208eaNe5fce09d.jpg', 'index': [u'beverage', u'food', u'lager beer', u'beer', u'brew', u'alcoholic beverage', u'bottle green color'], 'values': [0.954, 0.954, 0.755, 0.776, 0.781, 0.783, 1]}, '21': {'url': u'http://wx4.sinaimg.cn/mw690/006boOKDgy1fjggzqwijlj30j60j640w.jpg', 'index': [u'sectional furniture', u'furniture', u'indoors', u'sage green color'], 'values': [0.888, 0.97, 0.783, 0.92]}, '38': {'url': u'https://www.uooyoo.com/img2017/9/26/2017092663144657.jpg', 'index': [u'fabric', u'sage green color'], 'values': [0.819, 0.902]}, '42': {'url': u'http://k.zol-img.com.cn/diybbs/6080/a6079920.jpg', 'index': [u'desk', u'table', u'furniture', u'computer', u'microscope', u'sage green color'], 'values': [0.914, 0.914, 0.914, 0.799, 0.789, 0.915]}, '3': {'url': u'https://img.alicdn.com/imgextra/TB2T9B3Xg1J.eBjy0FaXXaXeVXa_!!1945434197.jpg', 'index': [u'arm', u'support', u'armchair', u'chair', u'seat', u'furniture', u'coal black color'], 'values': [0.867, 0.921, 0.759, 0.847, 0.854, 0.854, 0.929]}, '122': {'url': u'https://static1.paizi.com/uploadfile/2017/1013/20171013053252466.jpg', 'index': [u'gray color'], 'values': [0.761]}, '89': {'url': u'https://img14.360buyimg.com/imgzone/jfs/t175/247/2029997415/239022/6e87243d/53c0fa8eN843d8932.jpg', 'index': [u'crown jewels', u'holding device', u'headdress', u'alabaster color', u'ivory color'], 'values': [0.812, 0.786, 0.801, 0.934, 0.814]}, '125': {'url': u'https://img13.360buyimg.com/imgzone/jfs/t3439/263/728869669/71540/eca6cade/5811bb8aN791655b1.jpg', 'index': [u'cup', u'drinking vessel', u'coal black color'], 'values': [0.864, 0.809, 0.953]}, '93': {'url': u'http://www.bvh.cc/images/200912/goods_img/547_P_1260577161314.jpg', 'index': [u'floor lamp', u'lamp', u'furniture'], 'values': [0.992, 0.994, 0.994]}, '92': {'url': u'https://img.alicdn.com/imgextra/TB2VGxwd3vD8KJjy0FlXXagBFXa_!!2529740865.jpg', 'index': [u'percale (fabric)', u'fabric', u'claret red color'], 'values': [0.8, 0.8, 0.942]}}

l = label_df['index']
x = l[0]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[1]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[3]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[4]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[5]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[7]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);
x = l[9]
y = results[x]
display(HTML("<div style='margin-left:100px'><h3>Label: "+x+"</h3><br><img src='"+y['url']+"' width=200 height=200></div>"))
sns.barplot(y=y['index'], x=y['values']);