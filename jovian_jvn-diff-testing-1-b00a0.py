import jovian

import time
jovian.commit(environment=None)
for i in range(0,10):

    jovian.commit(environment=None, project="diff-testing-1")

    time.sleep(1)