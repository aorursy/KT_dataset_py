import pandas as pd

from glob import glob



# Get those with markings

# Credit https://www.kaggle.com/rohitsingh9990

markings_path = "../input/marker-images/marker_images/"

img_ext = 'png'

glob_obj = glob('{}*.{}'.format(markings_path,img_ext))

marker_ids = [file.split("/")[-1].strip(".png") 

              for file in glob_obj]

marker_dict = [{"image_id":img_id, "reason":"marks"} 

               for img_id in marker_ids]

marker_df = pd.DataFrame(marker_dict)



# Get those that are suspicious for mask/classification

# Credit https://www.kaggle.com/iamleonie

mask_df = pd.read_csv("../input/panda-suspicious-slides/suspicious_test_cases.csv")

mask_df.columns = ["image_id", "reason"]



# Get those that are blank

# Credit https://www.kaggle.com/yuvaramsing

blank_slides = ["3790f55cad63053e956fb73027179707"]

blank_masks = ["4a2ca53f240932e46eaf8959cb3f490a", 

               "3790f55cad63053e956fb73027179707", 

               "e4215cfc8c41ec04a55431cc413688a9", 

               "aaa5732cd49bffddf0d2b7d36fbb0a83"]

blanks = blank_slides + blank_masks

blanks_dict = [{"image_id":img_id, "reason":"blank"} 

               for img_id in blanks]

blanks_df = pd.DataFrame(blanks_dict)



#Get those with <3% tissue

# Credit https://www.kaggle.com/dannellyz

low_tiss_slides = [ '033e39459301e97e457232780a314ab7',

                    '0b6e34bf65ee0810c1a4bf702b667c88',

                    '3385a0f7f4f3e7e7b380325582b115c9',

                    '3790f55cad63053e956fb73027179707',

                    '5204134e82ce75b1109cc1913d81abc6',

                    'a08e24cff451d628df797efc4343e13c']

low_tiss_dict = [{"image_id":img_id, "reason":"tiss"} 

                 for img_id in low_tiss_slides]

tiss_df = pd.DataFrame(low_tiss_dict)



#Combine

all_suspicious = pd.concat([marker_df, mask_df, tiss_df, blanks_df])

print(f"There are a total of {len(all_suspicious)} slides.\nBreakdown:")

print(all_suspicious.reason.value_counts())

all_suspicious.to_csv("PANDA_Suspicious_Slides.csv", index=False)

all_suspicious.head()
class PandaData:

    """

    Description

    class for loading and sampling the PANDA dataset

    __________

    

    init:

    pct: float

        percentage of the dataset to sample

    cat_col: str

        this is the name of the column to sample equally from 

        based on observations. If this is left as none then a 

        simple pd.DataFrame.sample() is used

    sample: bool

        if false then sampling will not take place and instead

        the desired percentage will be returned with 

        pd.DataFrame.head()

    """

    

    #Establish Globals

    SLIDE_DIR = "../input/prostate-cancer-grade-assessment/train_images/"

    MASK_DIR = "../input/prostate-cancer-grade-assessment/train_label_masks/"

    TRAIN_LOC = "../input/prostate-cancer-grade-assessment/train.csv"

    SKIP_LOC = "../input/panda-analysis/PANDA_Suspicious_Slides.csv"

    

    def __init__(self, pct=.1, cat_col=None, sample=True):

        #Set up init variables

        self.cat_col = cat_col

        start_df = pd.read_csv(PandaData.TRAIN_LOC)

        skip_df = pd.read_csv(PandaData.SKIP_LOC)

        skip_ids = list(skip_df["image_id"])

        start_df = start_df[~start_df["image_id"].isin(skip_ids)]

        

        #Take percentages and turn into number

        self.n_sample = int(len(start_df) * pct)

        

        #Get subsample if desired

        if pct < 1:

            if sample:

                self.train_df = self.categorical_sample(start_df,

                                                       self.cat_col,

                                                       self.n_sample)

            else:

                self.train_df = start_df.head(self.n_sample)

        else:

            self.train_df = start_df.copy()

        

        #Clean up

        del start_df

        del skip_df

                

        

    def categorical_sample(self, df, cat_col, N):

        """

        Description

        __________

        sample evenly based on observations from categorical column

       



        Parameters

        __________

        df: Panda DataFrame

            df to sample from

        cat_col: str

            name of categorical to pull observations from

        N: int

            number to return from sample

        

        Returns (1)

        __________

        - Sampled Pandas DataFrame

        """

        

        if cat_col in df.columns:

            group = df.groupby(cat_col, group_keys=False)

            sample_df = group.apply(lambda g: g.sample(int(N * len(g) / len(df))))

        else:

            print("Col not found in df. Normal sample used.")

            sample_df = df.sample(N)

        return sample_df

    

#Sample run

panda_data = PandaData(pct=.1, cat_col="isup_grade", sample=True)

print(f"Total number sampled: {panda_data.n_sample}")

panda_data.train_df.head()