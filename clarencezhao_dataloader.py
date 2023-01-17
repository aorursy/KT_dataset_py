#Below is my recommended dataloader
#the dataloader object which loads the training images and gives them a label 
class DrivingDataset(Dataset):
    def __init__(self,data_dir, input_w=224, input_h=224,is_train=True,transform=None):
        if is_train==False:
            threshold=55#use 50 from each class as validation
        else:
            threshold=450#300 as training
        namelist = [0 for i in range(16)]# 15 list to contain images for each class
        
        self.data_filenames = []
        self.data_ids = []
        self.is_train=is_train

        self.data_root=fs.open_fs(data_dir)
        self.transform = transform
        keyword=['zero','one','two','three','four','five','six','seven','eight','nine','plus','minus','times','div','equal','decimal']
        for p in self.data_root.walk.files(filter=["*.jpg","*.png"]):
            filename=data_dir+p
            if is_train==True or 1==1:#temporary bug if in training, a label will be given to a training image depending on its folder name
                #like all images of 4 is contained in the folder "four"
                for i,j in enumerate(keyword):
                    if j in filename:
                        if namelist[i]<threshold:
                            self.data_filenames.append(filename)
                            self.data_ids.append(i)
                            namelist[i]+=1
                
            else:#if not training, it is not necessary to load a label
                self.data_filenames.append(filename)
                #self.data_ids.append(0)
        
        
        # print(self.data_filenames)
        #print(namelist)
        print(len(self.data_filenames))#displays how many images are there in a class


    def __getitem__(self, item):
        """Grey(i, j) = 0.299 × R(i, j) + 0.587 × G(i, j) + 0.114 × B(i, j)"""

        img_path = self.data_filenames[item]
        #print(img_path)
        target = self.data_ids[item]

        image = cv2.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        target = np.array([target], dtype=np.long)
        target = torch.from_numpy(target)
        
        return image,target

    def __len__(self):
        return len(self.data_filenames)
    