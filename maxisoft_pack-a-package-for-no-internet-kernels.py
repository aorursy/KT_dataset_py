! wget https://github.com/cornellius-gp/gpytorch/archive/v0.3.5.zip
! unzip v0.3.5.zip
from pathlib import Path

import shutil



non_required_files = (

    ".conda",

    ".github",

    "docs",

    "examples",

    "test",

    "environment.yml",

    "LICENSE",

    "pyproject.toml",

    "readthedocs.yml",

)



src_folder = Path("gpytorch-0.3.5")

for fname in non_required_files:

    target = src_folder / fname

    if target.is_dir():

        shutil.rmtree(target)

    else:

        target.unlink()
! ls -lh {src_folder}
import tarfile

import base64

import io





def pack_folder(target):

    buff = io.BytesIO()

    with tarfile.open(fileobj=buff, mode='w:xz') as tar:

        tar.add(target, arcname='.', recursive=True)



    buff.seek(0)

    return base64.b85encode(buff.getbuffer()) 





data = pack_folder(src_folder)

# data variable contains the encoded bytes





#print(data)
print(len(data))
from IPython.core.display import HTML

import html



data_id = str(hash(data)).zfill(8)[-8:]

escaped = html.escape(str(data))

js = """

function copy_cb(hash) {

  var inp = document.getElementById("i" + hash)

  inp.select()

  inp.setSelectionRange(0, inp.value.length + 1)

  document.execCommand('copy')

}"""



display(HTML(f"""

<script type="text/javascript">

{js}

</script>

<div name="resultstring" id="resultstring">

    <input id="i{data_id}" type="text" value="{escaped}"/>

    <button onclick="copy_cb({data_id})">Copy to clipboard</button>

</div>"""))
# cleanup

! rm -r gpytorch-0.3.5

! rm v0.3.5.zip