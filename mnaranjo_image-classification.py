import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # access big query tables
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# loading in big query table
data = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "eclipse_megamovie")
## check the tables included
## I want to look at photos v03 which includes all of images in jpg format
data.list_tables()
## see what the table looks like
## note that the totality and vision_labels are not necessarily accurate
data.head('photos_v_0_3')
# writing query and checking size
query = """    
SELECT id, jpeg_storage_uri
    FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_3` 
    """
data.estimate_query_size(query)
## For the moment, all I care about is where the image is stored
images = data.query_to_pandas_safe(query)
images.head()
# you always start in the working folder in kaggle
! ls
# lets back out of this and navigate to our code 
! cd ..
! ls
!cd input
!ls
!python ../input/eclipse-megamovie/label_image.py --graph=../input/eclipse-megamovie/retrained_graph.pb --labels=../input/eclipse-megamovie/retrained_labels.txt --image=../input/tf-files/ffe1a5ffd97ace6cc1a9f4e6c3edd50c138eb8ca7aa2337e87b0f55da4b70c13.jpg
# display the test image we just checked
img=mpimg.imread('../input/tf-files/ffe1a5ffd97ace6cc1a9f4e6c3edd50c138eb8ca7aa2337e87b0f55da4b70c13.jpg')
plt.figure(figsize=(10,10))
imgplot = plt.imshow(img)
plt.show()