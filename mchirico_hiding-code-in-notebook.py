

from IPython.display import HTML

htmlCodeHide="""

<style>

.button {

    background-color: #008CBA;;

    border: none;

    color: white;

    padding: 8px 22px;

    text-align: center;

    text-decoration: none;

    display: inline-block;

    font-size: 16px;

    margin: 4px 2px;

    cursor: pointer;

}

</style>

 <script>

   var divTag = document.getElementsByClassName("input")[0]

   var displaySetting = divTag.style.display;

   divTag.style.display = 'block';

 

    function toggleInput(i) { 

      var divTag = document.getElementsByClassName("input")[i]

      var displaySetting = divTag.style.display;

     

      if (displaySetting == 'block') { 

         divTag.style.display = 'none';

       }

      else { 

         divTag.style.display = 'block';

       } 

  }  

  </script>

  <button onclick="javascript:toggleInput(0)" class="button">Hide Code</button>

"""



print("Sample Output")



# HTML(""" <button onclick="javascript:toggleInput(0)" class="button">Hide Code</button>""")

HTML(htmlCodeHide)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



htmlCodeHide="""

 <script>

   var divTag = document.getElementsByClassName("input")[1]

   var displaySetting = divTag.style.display;

   divTag.style.display = 'block';

   </script>

  <button onclick="javascript:toggleInput(1)" class="button">Hide Code</button>

"""



print("Sample Output")



#HTML(htmlCodeHide)

HTML(""" <button onclick="javascript:toggleInput(1)" class="button">Hide Code</button>""")
print(check_output(["ls", "../input"]).decode("utf8"))

HTML(""" <button onclick="javascript:toggleInput(2)" class="button">Hide Code</button>""")


# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output





print("sample output 3")





HTML("""

 <script>

   var divTag = document.getElementsByClassName("input")[3]

   var displaySetting = divTag.style.display;

   divTag.style.display = 'block';

   </script>

  <button onclick="javascript:toggleInput(3)" class="button">Hide Code</button>

""")
print("stuff")

#HTML(htmlCodeHide)

HTML("""

 <script>

   var divTag = document.getElementsByClassName("input")[4]

   var displaySetting = divTag.style.display;

   divTag.style.display = 'none';

   </script>

  <button onclick="javascript:toggleInput(4)" class="button">Hide Code</button>

""")

print("The print will probably be hidden")



HTML("""

 <script>

   var divTag = document.getElementsByClassName("input")[5]

   var displaySetting = divTag.style.display;

   divTag.style.display = 'block';

   </script>

  <button onclick="javascript:toggleInput(5)" class="button">Toggle Code</button>

""")



# Hack to add space for hidden data

s="<br>"*10000

HTML(s)