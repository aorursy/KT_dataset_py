# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def My_TTA(model, TTA=3, batch_size=128, threshold = 0.1):

    

    model.eval()

    avg_predictions = {}

    ans_dict = {}

    models_num = len(os.listdir("../input/your_models_folder"))

    

    for time in range(TTA):

        

        test_transformed_dataset = iMetDataset(csv_file='sample_submission.csv', 

                                      label_file="labels.csv", 

                                      img_path="test/", 

                                      root_dir='../input/imet-2019-fgvc6/',

                                      transform=transforms.Compose([

                                          #

                                          # some data augumentations here

                                          #

                                          transforms.ToTensor(),

                                          transforms.Normalize(

                                              [0.485, 0.456, 0.406], 

                                              [0.229, 0.224, 0.225])

                                      ]))



        test_loader = DataLoader(

        test_transformed_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=8)



        with torch.no_grad():

            

            for i in range(models_num):



                model.load_state_dict(torch.load("../input/your_models_folder/your_model" + str(i)+ ".pth"))

   

                for batch_idx, sample in enumerate(test_loader):

     

                    image = sample["image"].to(device, dtype=torch.float)

                    img_ids = sample["img_id"]

                    predictions = model(image).cpu().numpy()

                    

                    for row, img_id in enumerate(img_ids):

                        if time == 0 and i == 0:

                            avg_predictions[img_id] = predictions[row]/(TTA*models_num)

                        else:

                            avg_predictions[img_id] += predictions[row]/(TTA*models_num)



                        if time == TTA - 1 and i == models_num -1:

                            all_class = np.nonzero(avg_predictions[img_id] > threshold)[0].tolist()

                            all_class = [str(x) for x in all_class]

                            ans_dict[img_id] = " ".join(all_class)

    

    return ans_dict