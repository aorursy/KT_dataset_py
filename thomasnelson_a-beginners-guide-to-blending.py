import pandas as pd



one = pd.read_csv('../input/champs-blending-tutorial/1.csv')

two = pd.read_csv('../input/champs-blending-tutorial/2.csv')

three = pd.read_csv('../input/champs-blending-tutorial/3.csv')



submission = pd.DataFrame()

submission['id'] = one.id

submission['scalar_coupling_constant'] = (0.40*one.scalar_coupling_constant) + (0.40*two.scalar_coupling_constant) + (0.20*three.scalar_coupling_constant)



submission.to_csv('super_blend2.csv', index=False)