import numpy as np
from numpy import *
from PIL import Image # Pillow : 이미지 처리 패키지
import matplotlib.pyplot as plt # matplotlib : 그래프 패키지
import matplotlib.patches as mpatches
import torch # torch, torchvision : 딥러닝 프레임워크
from torchvision import transforms, models
import cv2 # Open CV : 이미지 처리 패키지
model = models.segmentation.deeplabv3_resnet101(pretrained = True).eval()
labels = ['background', 'aeroplane', 'bicycle', 'bird','boat', 'bottle', 'bus',' car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

cmap = plt.cm.get_cmap('tab20c')
colors = (cmap(np.arange(cmap.N))*255).astype(np.int)[:, :3].tolist()
np.random.seed(2020)
np.random.shuffle(colors)
colors.insert(0, [0, 0, 0]) # background color must be black
colors = np.array(colors, dtype = np.uint8)

palette_map = np.empty((10, 0, 3), dtype = np.uint8)
legend = []

for i in range(21):
    legend.append(mpatches.Patch(color=np.array(colors[i])/255., label='%d: %s'%(i, labels[i])))
    c = np.full((10, 10, 3), colors[i], dtype = np.uint8)
    palette_map = np.concatenate([palette_map, c], axis=1)

plt.figure(figsize=(20, 2))
plt.legend(handles=legend)
plt.imshow(palette_map)
    
def segment(net, img):
    preprocess = transforms.Compose([ #torchvision.transform() : 이미지 전처리 모듈
        transforms.ToTensor(), # transforms.ToTensor() : 이미지를 텐서로 변환
        transforms.Normalize(  # transforms.Normalize() : 이미지 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # tenasor.unsqueeze() : 텐서에 차원(새로운 축)을 추가한다, squeeze의 반대
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda') # model(or batch).to('cuda') : 모델이나 배치를 CPU(RAM)로 부터 cuda 디바이스 메모리에 이동하여 처리한다
        model.to('cuda')
    
    output = model(input_batch)['out'][0] # (21, height, width)
    
    output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width) 
    # argmax(0) : 각 픽셀의 채널방향으로 최소값의 채널 인덱스를 반환
    # byte() : uint8 타입으로 변환 
    # cpu() : cuda -> cpu로 이동
    # numpy() : tensor -> numpy ndarray로 변환
    
    r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0])) 
    # Image.fromarray() : numpy ndarray를 Pillow Image 타입으로 반환
    # Image.resize() : 이미지 크기를 변형
    r.putpalette(colors)
    
    return r, output_predictions
img = np.array(Image.open('../input/imgs-mcp/02.jpg'))

fg_h, fg_w, _ = img.shape

segment_map, pred = segment(model, img)

fig, axes = plt.subplots(1, 2, figsize = (20, 10))
axes[0].imshow(img)
axes[1].imshow(segment_map)
