def calculate_metrics(self, outputs, targets, **kwargs):
    outputs = (outputs>0).float()
    targets = targets.view(-1,128,128)
    # This is different from the Dice logic
    # calculating intersection and union for a batch
    intersect = (outputs*targets).sum(2).sum(1)
    union = (outputs+targets).sum(2).sum(1)
    # Calculates the IOU, 0.001 makes sure the iou is 1 in case intersect
    # and union are both zero (where mask is zero and predicted mask is zero) -
    # this is a case of perfect match as well.
    iou = (intersect+0.001)/(union-intersect+0.001)
    # This simple logic here will give the correct result for precision
    # without going thru each threshold
    classification_precision = ((iou-0.451)*2*10).floor()/10
    # makes any ious less than 0.451 zero as well
    classification_precision[classification_precision<0] = 0
    # If you don't want the mean for the batch, you can return a list
    # of the classification_precision as well.  
    classification_precision = classification_precision.mean()
    return classification_precision