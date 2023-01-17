!git clone --depth 1 --branch "v1.0-alpha" https://github.com/crunchiness/lernd.git

%cd lernd

!pip install ordered-set==4.0.1
import os



import tensorflow as tf



from lernd.classes import GroundAtoms, ILP, LanguageModel, MaybeGroundAtom, ProgramTemplate

from lernd.lernd_loss import Lernd

from lernd.lernd_types import Constant, RuleTemplate

from lernd.main import generate_weight_matrices, extract_definitions, print_valuations

from lernd.util import ground_atom2str, str2ground_atom, str2pred, get_ground_atom_probs



os.environ['CUDA_VISIBLE_DEVICES'] = ''



target_pred = str2pred('predecessor/2')

zero_pred = str2pred('zero/1')

succ_pred = str2pred('succ/2')

preds_ext = [zero_pred, succ_pred]

constants = [Constant(str(i)) for i in range(10)]

language_model = LanguageModel(target_pred, preds_ext, constants)
preds_aux = []

rules = {target_pred: (RuleTemplate(0, False), None)}

forward_chaining_steps = 1

program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)
ground_zero = str2ground_atom('zero(0)')

background_axioms = [ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(9)]

positive_examples = [str2ground_atom(f'predecessor({i + 1},{i})') for i in range(9)]

print('Background axioms:')

print('\n'.join(map(ground_atom2str, background_axioms)))

print('\nPositive examples:')

print('\n'.join(map(ground_atom2str, positive_examples)))
ground_atoms = GroundAtoms(language_model, program_template)

negative_examples = []

for ground_atom, _ in ground_atoms.ground_atom_generator(MaybeGroundAtom.from_pred(target_pred)):

    if ground_atom not in positive_examples:

        negative_examples.append(ground_atom)

print('Negative examples:')

print('\n'.join(map(ground_atom2str, negative_examples)))
ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)
lernd_model = Lernd(ilp_problem, program_template)
weights = generate_weight_matrices(lernd_model.clauses)

losses = []

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.5)



im = None

for i in range(1, 301):

    loss_grad, loss, valuation, _ = lernd_model.grad(weights)

    optimizer.apply_gradients(zip(loss_grad, list(weights.values())))

    loss_float = float(loss.numpy())

    losses.append(loss_float)

    if i % 10 == 0:

        print(f'Step {i} loss: {loss_float}\n')
from matplotlib import pyplot as plt



fig, ax = plt.subplots()

ax.plot(losses)

ax.set_title('Loss')

ax.set_xlabel('Step')

ax.set_ylabel('Value')
extract_definitions(lernd_model.clauses, weights)

ground_atom_probs = get_ground_atom_probs(valuation, lernd_model.ground_atoms)

print_valuations(ground_atom_probs)