!git clone --depth 1 --branch "v1.0-alpha" https://github.com/crunchiness/lernd.git

%cd lernd

!pip install ordered-set==4.0.1
import os



import numpy as np

import tensorflow as tf

from IPython.core.display import clear_output

from matplotlib import pyplot as plt



from lernd.classes import ILP, LanguageModel, ProgramTemplate

from lernd.lernd_loss import Lernd

from lernd.lernd_types import Constant, RuleTemplate

from lernd.main import generate_weight_matrices, extract_definitions, print_valuations

from lernd.util import get_ground_atom_probs, ground_atom2str, softmax, str2ground_atom, str2pred



os.environ['CUDA_VISIBLE_DEVICES'] = ''



target_pred = str2pred('even/1')

zero_pred = str2pred('zero/1')

succ_pred = str2pred('succ/2')

preds_ext = [zero_pred, succ_pred]

constants = [Constant(str(i)) for i in range(11)]

language_model = LanguageModel(target_pred, preds_ext, constants)



# Program template

aux_pred = str2pred('pred/2')

aux_preds = [aux_pred]

rules = {

    target_pred: (RuleTemplate(0, False), RuleTemplate(1, True)),

    aux_pred: (RuleTemplate(1, False), None)

}

forward_chaining_steps = 6

program_template = ProgramTemplate(aux_preds, rules, forward_chaining_steps)



# ILP problem

ground_zero = str2ground_atom('zero(0)')

background = [ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(10)]

positive = [str2ground_atom(f'even({i})') for i in range(0, 11, 2)]

negative = [str2ground_atom(f'even({i})') for i in range(1, 10, 2)]

ilp_problem = ILP(language_model, background, positive, negative)
print('Background axioms:')

print(', '.join(map(ground_atom2str, background)))



print('\nPositive examples:')

print(', '.join(map(ground_atom2str, positive)))



print('\nNegative examples:')

print(', '.join(map(ground_atom2str, negative)))
lernd_model = Lernd(ilp_problem, program_template, mini_batch=0.3)

weights = generate_weight_matrices(lernd_model.clauses)



losses = []

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.5)



for i in range(1, 501):

    loss_grad, loss, valuation, full_loss = lernd_model.grad(weights)

    optimizer.apply_gradients(zip(loss_grad, list(weights.values())))

    loss_float = float(full_loss.numpy())

    mb_loss_float = float(loss.numpy())

    losses.append(loss_float)

    if i % 10 == 0:

        print(f'Step {i} loss: {loss_float}, mini_batch loss: {mb_loss_float}\n')

        fig, axs = plt.subplots(ncols=3, gridspec_kw={'width_ratios': [1, 3, 0.2]})

        fig.subplots_adjust(top=0.8, wspace=0.6)

        fig.suptitle(f'Softmaxed weight matrices at step {i}', fontsize=16)

        im0 = axs[0].pcolormesh(softmax(weights[aux_pred]).numpy(), cmap='viridis', vmin=0, vmax=1)

        axs[0].set_title('Auxiliary predicate')

        im1 = axs[1].pcolormesh(np.transpose(softmax(weights[target_pred]).numpy()), cmap='viridis', vmin=0, vmax=1)

        axs[1].set_title('Target predicate')

        fig.colorbar(im0, cax=axs[2])

        plt.show()

        if i != 500:

            clear_output(wait=True)
fig, ax = plt.subplots()

ax.plot(losses)

ax.set_title('Loss')

ax.set_xlabel('Step')

ax.set_ylabel('Value')
extract_definitions(lernd_model.clauses, weights)

ground_atom_probs = get_ground_atom_probs(valuation, lernd_model.ground_atoms)

print_valuations(ground_atom_probs)