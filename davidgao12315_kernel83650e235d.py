%matplotlib inline
from skimage import io,color,exposure,morphology,transform,img_as_float
import matplotlib.pyplot as plt
import skimage.transform as st
import numpy as np
import os
from PIL import Image
import imagehash
import shutil
import cv2 as cv
import time
import copy
def tagRMB(imgPath,tpl):
    target=cv.imread(imgPath)
    
    th,tw=tpl.shape[:2]
#     res=cv.matchTemplate(target,tpl,cv.TM_CCOEFF_NORMED)
    res=cv.matchTemplate(target,tpl,cv.TM_CCOEFF)

    # axis=1,行方向
    res_sorted=sorted(res.max(axis=1),reverse=True)
    res_dif=[0]*150
    for i in range(150):
        res_dif[i]=(res_sorted[i]-res_sorted[i+1])*100./res_sorted[i+1]

    max_lastIdx=res_dif.index(sorted(res_dif,reverse=True)[0])

    idx=np.argwhere(res >= res_sorted[max_lastIdx])
    idx_set=set(np.unique(idx[:,0]))

    for i in range(len(idx)):
        if idx[i,0] in idx_set:
            idx_set.remove(idx[i,0])
            tl=(idx[i,1],idx[i,0])
            br=(tl[0]+tw,tl[1]+th)
            cv.rectangle(target,tl,br,(0,0,0),1)

    cv.imwrite(imgPath,target)    
# 膨胀腐蚀操作
def dil2ero(img,selem):
    img=morphology.dilation(img,selem)
    imgres=morphology.erosion(img,selem)
    return imgres


# 收缩点团为单像素点（3×3）
def isolate(img):
    idx=np.argwhere(img<1)
    rows,cols=img.shape
    
    for i in range(idx.shape[0]):
        c_row=idx[i,0]
        c_col=idx[i,1]
        if c_col+1<cols and c_row+1<rows:
            img[c_row,c_col+1]=1
            img[c_row+1,c_col]=1
            img[c_row+1,c_col+1]=1
        if c_col+2<cols and c_row+2<rows:
            img[c_row+1,c_col+2]=1
            img[c_row+2,c_col]=1
            img[c_row,c_col+2]=1
            img[c_row+2,c_col+1]=1
            img[c_row+2,c_col+2]=1
    return img


# 将图像边框变白
def clearEdge(img,width):
    img[0:width-1,:]=1
    img[1-width:-1,:]=1
    img[:,0:width-1]=1
    img[:,1-width:-1]=1
    return img


# 创建文件夹
def mkdir(path):
    import os
    
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('文件夹创建成功')
        return True
    else:
        print('文件夹已存在')
        shutil.rmtree(path)
        os.makedirs(path)
        print('已清空文件夹')
        return False
def splitImg(imgFilePath,split_pic_path):
    '''
    imgFilePath，图片路径
    split_pic_path，目标文件夹路径
    '''
    shutil.rmtree(tempDir)
    mkdir(tempDir)
    
    # 读取图片，并转灰度图
    img=io.imread(imgFilePath,True)
    #二值化
    img_forSplit=copy.deepcopy(img)
    #img 提取边框用
    bi_th=img.max()*0.875
    img[img<bi_th]=0
    img[img>=bi_th]=1
    # img_forSplit 分割用
    bi_th=0.733
    img_forSplit[img_forSplit<bi_th]=0
    img_forSplit[img_forSplit>=bi_th]=1
    
    #######################################################
    # 求图像中的横线和竖线
    rows,cols=img.shape
    scale=80

    col_selem=morphology.rectangle(cols//scale,1)
    img_cols=dil2ero(img,col_selem)
    # io.imsave(deskpath+'img_cols.jpg',img_cols)

    row_selem=morphology.rectangle(1,rows//scale)
    img_rows=dil2ero(img,row_selem)
    # io.imsave(deskpath+'img_rows.jpg',img_rows)

    ########################################################
    img_line=img_cols*img_rows
    # io.imsave(deskpath+'img_line.jpg',img_line)

    # col和row的图里的线有点短。。。求dot图的时候刚好没重叠。。
    # 先延长，再求dot图
    idx_img_rows=np.argwhere(img_rows==0)
    img_rows_temp=img_rows
    for i in range(idx_img_rows.shape[0]):
        img_rows_temp[
            idx_img_rows[i,0],
            idx_img_rows[i,1]+1 if idx_img_rows[i,1]+1<cols else idx_img_rows[i,1]]=0

    img_dot=img_cols+img_rows_temp
    img_dot[img_dot>0]=1
    img_dot=clearEdge(img_dot,3)
    img_dot=isolate(img_dot)
    # io.imsave(deskpath+'img_dot.jpg',img_dot)

    ########################################################
    # mkpath=deskpath+'input\\'
    # mkdir(mkpath)

    #获取表格顶点位置idx
    idx=np.argwhere(img_dot<1)
    idx_unirow=np.unique(idx[:,0])
    #一行一行的来处理各个点
    
    # 保存计数器
    countHere=0
    for i in range(idx_unirow.shape[0]-1):
        #当前行号、下一行行号、中间行号
        r_cur=idx_unirow[i]
        r_next=idx_unirow[i+1]
        r_mid=(r_cur+r_next)//2

        idx_currow=idx[idx[:,0]==r_cur]
        idx_nextrow=idx[idx[:,0]==r_next]

        #遍历当前行的前n-1个点
        for j in range(idx_currow.shape[0]-1): 
            #当左上角顶点下没有line的时候，则不是一个单元格的起始顶点
            if(img_line[r_mid,idx_currow[j,1]]==1):
                continue


            offset=1
            bottom_c=0
            while(j+offset<idx_currow.shape[0]):
                # 找单元格的右上角顶点
                if(img_line[r_mid,idx_currow[j+offset,1]]==1):
                    offset=offset+1
                else:
                    bottom_c=idx_currow[j+offset,1]
                    break

            if bottom_c==0:
                continue

            idx_temp=idx_nextrow[idx_nextrow[:,1]==bottom_c]
            if (idx_temp.shape[0]>0):
                imghere=img_forSplit[r_cur:r_next,idx_currow[j,1]:bottom_c]
                countHere+=1
                io.imsave(split_pic_path+'\\'+'{0:0>4}_{1:0>6}'.format(i,countHere)+'.png',imghere)
def focusImg(imgPath):
    img=io.imread(imgPath)
    img=color.rgb2gray(img)
    img=img_as_float(img)
    img=clearEdge(img,3)

    # 求各列的和
    col_sum=img.sum(axis=0)
    # 求各行的和
    row_sum=img.sum(axis=1)

    idx_col_sum=np.argwhere(col_sum<col_sum.max())
    if len(idx_col_sum)==0:
        os.remove(imgPath)
        return
    col_start,col_end=idx_col_sum[0,0]-1,idx_col_sum[-1,0]+2
    
    idx_row_sum=np.argwhere(row_sum<row_sum.max())
    if len(idx_row_sum)==0:
        os.remove(imgPath)
        return
    row_start,row_end=idx_row_sum[0,0]-1,idx_row_sum[-1,0]+2
    
    # 覆盖源文件保存
    io.imsave(imgPath,img[row_start:row_end,col_start:col_end])
#     sizeStr=str(row_end-row_start)+"×"+str(col_end-col_start)
#     return sizeStr
def splitChar(img):
    imgF=img_as_float(img)
    # 求各列的和
    col_sum=imgF.sum(axis=0)
    idx=np.argwhere(col_sum==col_sum.max())

    images=[]
    countHere=0
    for i in range(1,len(idx)):
        if idx[i,0]-idx[i-1,0]>1:
            countHere+=1
            # io.imsave(saveDir+str(countHere)+'_'+fname+'.png',img[:,idx[i-1,0]:idx[i,0]+1])
            imgHere=img.crop((idx[i-1,0],0,idx[i,0]+1,img.height))
            # imgHere=imgF[:,idx[i-1,0]:idx[i,0]+1]
            #[:,idx[i-1,0]:idx[i,0]+1]
            images.append(imgHere)
    
    return images
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable

# dicRes={
#     0:'0',
#     1:'1',
#     2:'2',       
#     3:'3',
#     4:'4',
#     5:'5',       
#     6:'6',
#     7:'7',
#     8:'8',       
#     9:'9',
#     10:'N',
#     11:',',     
#     12:'.',
#     13:'—',
#     14:'/',    
#     15:'*',
#     16:'无'           
# }

# def softmaxPred(data):
#     '''data是一个[1,1,28,28]的图'''
#     data=torch.FloatTensor(data)
#     data=Variable(data, volatile=True)
#     output = model(data)
#     # get the index of the max
#     pred = output.data.max(1, keepdim=True)[1]
#     return dicRes[int(pred)]


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.l1 = nn.Linear(784, 520)
#         self.l2 = nn.Linear(520, 320)
#         self.l3 = nn.Linear(320, 240)
#         self.l4 = nn.Linear(240, 120)
#         self.l5 = nn.Linear(120, 17)

#     def forward(self, x):
#         # Flatten the data (n, 1, 28, 28) --> (n, 784)
#         x = x.view(-1, 784)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         return F.log_softmax(self.l5(x), dim=1)

# model = Net()
# # 加载已训练好的参数
# model.load_state_dict(torch.load('my_params.pkl'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

dicRes={
    0:'0',
    1:'1',
    2:'2',       
    3:'3',
    4:'4',
    5:'5',       
    6:'6',
    7:'7',
    8:'8',       
    9:'9',
    10:'N',
    11:',',     
    12:'.',
    13:'—',
    14:'/',    
    15:'*',
    16:'无'           
}

def cnnPred(data):
    '''data是一个[1,1,28,28]的图'''
    data.resize((1,1,28,28))
    data=torch.FloatTensor(data)
    data=Variable(data, volatile=True)
    output = model(data)
    # get the index of the max
    pred = output.data.max(1, keepdim=True)[1]
    return dicRes[int(pred)]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #1*28*28
        self.conv1=nn.Sequential(
            #16*28*28
            #padding=(ks-1)/2时，图像大小不变
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            #16*14*14
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            #32*14*14
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            #32*7*7
            nn.MaxPool2d(2)
        )
        self.out=nn.Linear(32*7*7,17)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

model = CNN()
# 加载已训练好的参数
model.load_state_dict(torch.load('my_CNN_params.pkl'))
hash_size=20
phashHamming_th=100

def hamming(h1, h2):
    '''计算两图的汉明距离'''
    return sum(sum(h1.hash^h2.hash))


def getText(img,bigPicTempleDic):
    hashHere=imagehash.phash(img,hash_size=hash_size)
    seq=[]
    for key in bigPicTempleDic.keys():
        seq.append((hamming(hashHere,bigPicTempleDic[key]),key))
        
    res_minDis,res_key=sorted(seq,key=lambda x:x[0])[0]
    #print('HMdis:{0}'.format(res_minDis))
    if res_minDis<phashHamming_th:
        return res_key
    else:
        return '未识别出来'
templePath=r'D:\Desktop\bigPicTemple'

bigPicTempleDic={'keyname':'phash'}
# 初始化模板字典
# dict{文件名，phash}
for fpath,fdir,fs in os.walk(templePath):
    for f in fs:
        fname,fext=os.path.splitext(f)
        bigPicTempleDic[fname]=imagehash.phash(
            Image.open(os.path.join(fpath,f)),
            hash_size=hash_size)
        
del bigPicTempleDic['keyname']
def my_main(srcFilePath,txtpath,fnameIn,tempdir):
    starttime=time.clock()

    print('标记人民币账户')
    tagRMB(srcFilePath,tpl)

    print('开始分割单元格')
    # 分割单元格，二值化
    splitImg(srcFilePath,tempdir)

    print('开始focus')
    # focus，覆盖原图
    for fpath,fdir,fs in os.walk(tempdir):
        for f in fs:
            filepath=os.path.join(fpath,f)
            focusImg(filepath)

    print('开始分类')
    with open(txtpath,'w') as ftxt:
        # 判断大小图，做相应处理，并将结果输出到txt
        for fpath,fdir,fs in os.walk(tempdir):
            for f in fs:
                filepath=os.path.join(fpath,f)
                fname,fext=os.path.splitext(f)
                img=Image.open(filepath)

                ftxt.write(fname+":")
                # 高度<= 13，是小图，缩softmax
                if img.size[1]<=13:
                    # 先分割字符
                    images=splitChar(img)
                    for i in range(len(images)):
                        imgSplit=images[i]
                        imgSplit=imgSplit.resize((28,28))
                        #res=softmaxPred(np.array(imgSplit))
                        res=cnnPred(np.array(imgSplit))
                        ftxt.write(res)
                        # print("file:{0}\nres_small:{1}\n".format(f,res))
                    ftxt.write('\n')
                else:
                    #做phash
                    res=getText(img,bigPicTempleDic)
                    ftxt.write(res+'\n')
                    # print("file:{0}\nres_big:{1}\n".format(f,res))

    endtime=time.clock()
    print(endtime-starttime)
    print('done\n')
tplpath=r'Data/input/biaoge/zhibiao.jpg'
tpl=cv.imread(tplpath)

tfileDir=r'D:\Desktop\tfile'
tempDir=r'D:\Desktop\temp'
res_txtDir=r'D:\Desktop\res_txt'

countf=0
for fpath,fdir,fs in os.walk(tfileDir):
    for f in fs:
        filepath=os.path.join(fpath,f)
        fname=os.path.splitext(f)[0]
        txtpath=os.path.join(res_txtDir,fname+'.txt')
        countf+=1
        print('index:{0},开始做{1}'.format(countf,f))
        my_main(filepath,txtpath,fname,tempDir)

print('\n跑完了')