import torch
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import time 
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
video = cv2.VideoCapture(input("video file"))
while(True):    
    ret, frame = vid.read() 
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
        
    output_predictions = output.argmax(0)
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    
    r.show()
    r.close()
    time.sleep(5)    