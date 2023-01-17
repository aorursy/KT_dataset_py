#Importando Libs

import tensorflow

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np

# Plot

import matplotlib.pyplot as plt

%matplotlib inline
# Carregando o modelo ResNet pré-treinado no ImageNet

model = ResNet50(weights='imagenet')
#Buscando alguma imagem da Internet

import requests



url = 'https://fotos.jornaldocarro.estadao.com.br/uploads/2020/01/03062028/Onix-2020-Hatch-2-1160x774.jpg'

url = 'https://quatrorodas.abril.com.br/wp-content/uploads/2019/04/imagem7-e1584036891780.jpg?quality=70&strip=info'

url = 'https://www.calimaro.com.br/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/s/h/shelter-violao-de-aco-auditorium-sag500prsx-nm-natural-matte-principal.jpg'

url = 'https://http2.mlstatic.com/lote-com-4-peixe-palhaco-ocelaris-com-brinde-marinho-D_NQ_NP_900038-MLB27109155440_032018-F.webp'

r = requests.get(url, allow_redirects=True)



open('test.jpg', 'wb').write(r.content)
# Abrindo arquivo salvo e botando no formato da Resnet

img_path = './test.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



preds = model.predict(x)

# Resultados como uma lista de (classid, descrição, probabilidade).

print('Predicted:', decode_predictions(preds, top=3)[0])
#Visualizando

import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(img)
#Mas como é esse modelo?

model.summary()