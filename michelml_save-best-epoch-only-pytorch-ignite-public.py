import os

import torch

from torchvision import models



from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Loss, Accuracy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet18(pretrained=True)



criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)



metrics = {

    'loss': Loss(criterion),

    'accuracy': Accuracy(),

}



trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
# create models directory if it doesn't exist

!mkdir -p models
def get_saved_model_path(epoch):

    return f'models/Model_{model_name}_{epoch}.pth'



best_acc = 0.

best_epoch = 1

best_epoch_file = ''



@trainer.on(Events.EPOCH_COMPLETED)

def save_best_epoch_only(engine):

    epoch = engine.state.epoch



    global best_acc

    global best_epoch

    global best_epoch_file

    best_acc = 0. if epoch == 1 else best_acc

    best_epoch = 1 if epoch == 1 else best_epoch

    best_epoch_file = '' if epoch == 1 else best_epoch_file



    metrics = val_evaluator.run(val_loader).metrics



    if metrics['accuracy'] > best_acc:

        prev_best_epoch_file = get_saved_model_path(best_epoch)

        if os.path.exists(prev_best_epoch_file):

            os.remove(prev_best_epoch_file)

            

        best_acc = metrics['accuracy']

        best_epoch = epoch

        best_epoch_file = get_saved_model_path(best_epoch)

        print(f'\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}\n\n\n')

        torch.save(model.state_dict(), best_epoch_file)