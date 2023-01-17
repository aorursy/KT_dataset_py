import jovian
# Pipe output to do basic analysis

!ls ../input/train/ | wc -l

!ls ../input/train/ | head
jovian.commit(environment=None)
# Use the full power of Python

test_ids = list(set([str(fn).split('/')[-1].split('_')[0]  for fn in TEST_DIR.iterdir()]))

print('Test IDs:', len(test_ids))

test_ids[:10]