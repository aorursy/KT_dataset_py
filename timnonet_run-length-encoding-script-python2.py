import numpy as np

import pandas as pd
# Create submission DataFrame

def create_submission(csv_name, predictions, image_ids):

    """

    csv_name -> string for csv ("XXXXXXX.csv")

    predictions -> numpyarray of size (num_examples, height, width)

                In this case (num_examples, 512, 512)

    image_ids -> numpyarray or list of size (num_examples,)

    

    predictions[i] should be the prediction of road for image_id[i]

    """

    sub = pd.DataFrame()

    sub['ImageId'] = image_ids

    encodings = []

    num_images = len(image_ids)

    for i in range(num_images):

        if (i+1) % (num_images//10) == 0:

            print(i, num_images)

        encodings.append(rle_encoding(predictions[i]))

        

    sub['EncodedPixels'] = encodings

    #sub['Height'] = [512]*num_images Nonger needed for DICE Scoring

    #sub['Width'] = [512]*num_images Nonger needed for DICE Scoring

    sub.to_csv(csv_name, index=False)



# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    """

    x = numpyarray of size (height, width) representing the mask of an image

    if x[i,j] == 0:

        image[i,j] is not a road pixel

    if x[i,j] != 0:

        image[i,j] is a road pixel

    """

    dots = np.where(x.T.flatten() != 0)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): 

            run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths