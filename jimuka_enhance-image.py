# 1.4 data augmentation
import imgaug as ia
from imgaug import augmenters as iaa


def augmentation_generator(yolo_dataset):
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                     y1=bb[1],
                                     x2=bb[2],
                                     y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] + bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)),  # change brightness
            # iaa.ContrastNormalization((0.5, 1.5)),
            # iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
        ])
        # seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i, j, 0] = bb.x1
                boxes[i, j, 1] = bb.y1
                boxes[i, j, 2] = bb.x2
                boxes[i, j, 3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        # batch = (img_aug, boxes)
        yield batch

aug_train_db = augmentation_generator(train_db)
db_visualize(aug_train_db)