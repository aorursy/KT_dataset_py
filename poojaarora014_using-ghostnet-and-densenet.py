import sys



import torch

import pandas as pd

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SequentialSampler

from torchvision.models.densenet import densenet121



sys.path.insert(0, "/kaggle/input/ghostnetbengali")



from ghost_net import ghost_net
IMAGE_SIZE = 137, 236





class DatasetRetriever(Dataset):

    

    def __init__(self, df):

        self.image_ids = df.iloc[:, 0].tolist()

        self.images = torch.from_numpy(255 - df[[str(x) for x in range(32332)]].values)



    def __len__(self):

        return self.images.shape[0]



    def __getitem__(self, idx):

        img = self.images[idx]

        img = img.view(*IMAGE_SIZE)

        img = img.to(torch.float32) / 255.

        img = img.unsqueeze(0)

        img = img.repeat(3, 1, 1)

        img_id = self.image_ids[idx]        

        return img_id, img

    

class Predictor:

    def __init__(self, model):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device, dtype=torch.float32);

        self.model.eval()

        print(f'Model prepared. Device is {self.device}')

    

    def predict(self, inputs):

        inputs = inputs.to(self.device, dtype=torch.float32)

        with torch.no_grad():

            outputs = self.model(inputs)

        return outputs



    def load(self, path):

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
predictors = []



for i, (model_path, net, coef) in enumerate([

    ('/kaggle/input/ghostnetbengali/checkpoint.pt', ghost_net(num_classes=168+11+7), 0.5),

    ('/kaggle/input/ghostnetbengali/densenet121-checkpoint.pt', densenet121(num_classes=168+11+7), 0.5),

]):

    predictor = Predictor(net)

    predictor.load(model_path)

    predictors.append((predictor, coef))





def predict_to_numpy(predict):

    return torch.nn.functional.softmax(predict, dim=1).data.cpu().numpy().argmax(axis=1)



def make_prediction(images):

    outputs = 0

    for predictor, coef in predictors:

        outputs += predictor.predict(images) * coef



    roots = predict_to_numpy(outputs[:,:168])

    vowels = predict_to_numpy(outputs[:,168:168+11])

    consonants = predict_to_numpy(outputs[:,168+11:])

    return roots, vowels, consonants
%%time



target = []

row_id = []



for i in range(4):

    data = pd.read_parquet(f'../input/bengaliai-cv19/test_image_data_{i}.parquet')

    dataset = DatasetRetriever(data)

    data_loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False, sampler=SequentialSampler(dataset))



    for idx, (image_ids, images) in tqdm(enumerate(data_loader), total=len(data_loader)):



        roots, vowels, consonants = make_prediction(images)



        for image_id, root, vowel, consonant in zip(image_ids, roots, vowels, consonants):

            row_id.append(image_id + '_consonant_diacritic')

            target.append(consonant)

            row_id.append(image_id + '_grapheme_root')

            target.append(root)

            row_id.append(image_id + '_vowel_diacritic')

            target.append(vowel)
df_submission = pd.DataFrame(

    {

        'row_id': row_id,

        'target': target

    },

    columns=['row_id','target']

)



df_submission.to_csv('submission.csv', index=False)



df_submission.head(10)