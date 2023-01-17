! pip install jieba
import jieba 
import pickle

content = input("请输入评论:")
print("预测内容:" + content)

with open('../input/models/count-vectorizer.pickle', 'rb') as fr:
    vectorizer = pickle.load(fr)
    
    
with open('../input/models/bow-model.pickle', 'rb') as fr:
    bow_model = pickle.load(fr)
    

def jiebax(x):
    x = str(x)
    bd = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． '
    for i in bd:
        x = x.replace(i, '')
    return ' '.join(jieba.cut(x, cut_all=False))

x = [jiebax(content)]
x = vectorizer.transform(x)

print("模型预测中...")
y_predict = bow_model.predict(x)

if y_predict[0] == 1:
    print("喜欢!")
else:
    print("不喜欢！")