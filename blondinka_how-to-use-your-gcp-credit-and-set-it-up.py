"""

#!/usr/bin/env bash



CUR_DIR=$pwd

DATA_DIR_LOC=dataset



mkdir -p $DATA_DIR_LOC

cd $DATA_DIR_LOC



if [ "$(ls -A $(pwd))" ]

then

    echo "$(pwd) not empty!"

else

    echo "$(pwd) is empty!"

    pip install kaggle --upgrade

    kaggle competitions download -c severstal-steel-defect-detection

    mkdir train

    mkdir test

    unzip train_images.zip -d train

    unzip test_images.zip -d test

fi



cd $CUR_DIR

echo $(pwd)

"""
print("Thank you")