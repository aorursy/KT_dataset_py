import pandas as pd

train_csv=pd.read_csv('../input/global-wheat-detection/train.csv')

# print(train_csv)

# print(train_csv['image_id'].values)



train_csv_set_list=list(set(train_csv['image_id'].values))

print(len(train_csv_set_list))



# 随机打乱数据

print(train_csv_set_list[:10])

from random import shuffle

shuffle(train_csv_set_list)

print(train_csv_set_list[:10])



train_data=train_csv_set_list[:2600]

val_data=train_csv_set_list[2600:]

print(len(train_data))

print(len(val_data))
def get_output_dict(train_csv_set_list): 

    #先生成categories：

    CATEGORIES_DICT=["id","name"]

    _CATEGORIES=["wheat",]

    def generate_categories_dict(category):

        return [{CATEGORIES_DICT[0]:category.index(x),CATEGORIES_DICT[1]:x} for x in category]

    categories_dict_list=generate_categories_dict(_CATEGORIES)

    #print(generate_categories_dict(_CATEGORIES))



    categories_name_to_id_dict = {}

    for x in _CATEGORIES:

        categories_name_to_id_dict[x]=_CATEGORIES.index(x)

    # print(categories_name_to_id_dict)







    # 生成"images","annotations"

    images_dict_list=[]

    annotations_dict_list=[]

    id=0



    import cv2

    for i in range(len(train_csv_set_list)):    

        an_images_file_name=train_csv_set_list[i]+".jpg"

        # print(an_images_file_name)



        images_dict_list.append({"id": i,"width":1024,"height":1024,"file_name":an_images_file_name})

    # print(images_dict_list)



        # print(train_csv[train_csv.image_id == train_csv_set_list[i]])

        for j in train_csv[train_csv.image_id == train_csv_set_list[i]]['bbox'].values:

            #print(type(j))

            j=eval(j)

            #print(j)

            #print(type(j))





            temp_annotations_dict={"image_id":i,'area':1048576,"iscrowd":0,"bbox":j,"category_id":0,"id":id,"segmentation":[]}       

            id+=1



            annotations_dict_list.append(temp_annotations_dict)





    # 生成最终标签以及保存文件

    COCO_DICT=["images","annotations","categories"]









    output={COCO_DICT[0]:images_dict_list,COCO_DICT[1]:annotations_dict_list,COCO_DICT[2]:categories_dict_list}

    return output





train_output=get_output_dict(train_data)

val_output=get_output_dict(val_data)


import json

def save_json(dict,path):

    with open(path,"w") as f:

        json.dump(dict,f)

        

save_json(train_output,'instances_train2017.json')

save_json(val_output,'instances_val2017.json')